# H100 PCI Passthrough — Full Change Log & Permanent Fix Plan

## Environment

| Item | Value |
|------|-------|
| OpenStack release | Caracal (2024.1), Nova 29.4.0 |
| Kubernetes / GitOps | openstack-helm on k8s, FluxCD |
| nova namespace | `nova` (not `openstack`) |
| Nova image | `cr.gitlab-int.switch.ch/engines2/images/e2-custom-images/nova:2024.1-main.20251119-14.01.07` |
| libvirt | 8.0.0 |
| QEMU | 6.2.0, machine type `pc-q35-6.2` |
| Host / Guest OS | Ubuntu 24.04, kernel 6.8.0-52-generic |
| GPUs | 2× Nvidia H100L 94 GB (95 830 MiB each) |
| Host PCI addresses | `0000:ca:00.0`, `0000:e1:00.0` (vfio-pci) |
| Guest PCI addresses | `0000:05:00.0`, `0000:06:00.0` |
| GPU nova-compute DaemonSet | `nova-compute-intel-gpu-c2fd60e1` (hash stable unless label selector changes) |

---

## Problem Statement

### Symptom

Guest VM kernel log:
```
NVRM: This PCI I/O region assigned to your NVIDIA device is invalid
nvidia: probe of 0000:05:00.0 failed with error -1
```

### Root Cause (two compounding bugs)

1. **`hw:pcihole64` is NOT implemented in Nova 29.4.0.**
   The flavor extra_spec exists in documentation but Nova never reads it — it generates no pcihole64 configuration at all. The guest therefore gets QEMU's default 2 GB 64-bit PCI hole, which is far too small for two H100L cards (each needs ~128 GB BAR1).

2. **libvirt 8.0.0 silently drops `<target><pcihole64>` for q35 machines.**
   Even if Nova were to generate the correct libvirt XML, libvirt would silently strip it. The `<pcihole64>` element only works under `<pci-root>` (i440fx machines), not under `<pcie-root>` (q35 machines).

### Why 512 GB?

Each H100L 94 GB requires ~128 GB of 64-bit PCI address space (BAR1 = framebuffer).
2 GPUs × 256 GB rounded up to next power of two = **512 GB = 549 755 813 888 bytes**.

---

## Solution

For q35 machines the correct knob is a QEMU global property, not a libvirt XML element:

```
-global q35-pcihost.pci-hole64-size=549755813888
```

libvirt's `<qemu:commandline>` extension passes arbitrary QEMU args through without modification.
Nova is patched to emit this arg whenever `hw:pcihole64` is set in the flavor.

---

## Flavor

Name: `gpu-2H100L.c042r256` — 42 vCPUs, 256 GB RAM

```
aggregate_instance_extra_specs:gpu='true'
hw:iommu='True'
hw:pci_numa_affinity_policy='required'
hw:pcihole64='549755813888'
hw:vif_multiqueue_enabled='true'
pci_passthrough:alias='NVIDIA-H100-94G:2'
```

---

## Nova Patches (both files confirmed working)

### `nova/virt/libvirt/config.py`

Add `qemu_args` list to `LibvirtConfigGuest` and emit it as `<qemu:commandline>` in `format_dom()`.

```python
# In LibvirtConfigGuest.__init__  (after self.metadata = [])
self.qemu_args = []   # extra QEMU -global args injected via <qemu:commandline>

# In LibvirtConfigGuest.format_dom()  (just before `return root`)
if self.qemu_args:
    qemu_ns = 'http://libvirt.org/schemas/domain/qemu/1.0'
    cmdline = etree.SubElement(root, '{%s}commandline' % qemu_ns)
    for val in self.qemu_args:
        arg_el = etree.SubElement(cmdline, '{%s}arg' % qemu_ns)
        arg_el.set('value', val)
```

> **Important:** Do NOT use `root.set('xmlns:qemu', …)` — lxml raises
> `ValueError: Invalid attribute name`. Use clark notation `{ns}element`
> and lxml adds the namespace declaration automatically.
>
> **Important:** `LOG` is not in scope in `config.py` — do not add log
> statements there.

### `nova/virt/libvirt/driver.py`

At the `_guest_needs_pcie` call site, read `hw:pcihole64` from the flavor and
attach it to `guest.qemu_args`.

```python
_pcihole64 = flavor.extra_specs.get('hw:pcihole64')
LOG.warning('PCIHOLE64 extra_specs=%s val=%s', flavor.extra_specs, _pcihole64)
self._guest_add_pcie_root_ports(guest)
if _pcihole64:
    guest.qemu_args.extend(
        ['-global', 'q35-pcihost.pci-hole64-size=%s' % _pcihole64])
    LOG.warning('PCIHOLE64 injected qemu_args=%s', guest.qemu_args)
```

Both patches are **idempotent** — the patch script checks for `qemu_args` before
modifying the file.

---

## Current State (temporary, in-cluster only)

Applied manually against the live DaemonSet. Will be lost on the next Helm
reconciliation or DaemonSet rollout triggered by FluxCD.

| Resource | Namespace | Status |
|----------|-----------|--------|
| ConfigMap `nova-pcihole64-patch` | `nova` | exists in-cluster, NOT in git |
| DaemonSet `nova-compute-intel-gpu-c2fd60e1` patch | `nova` | exists in-cluster, NOT in git |

---

## Permanent Fix — Changes Required

### Overview

Apply the DaemonSet patch declaratively through FluxCD so it survives every
Helm reconciliation. Two possible approaches:

**Approach A — HelmRelease `postRenderers` (kustomize strategic merge patch)**
Add a `postRenderers` block to the nova `HelmRelease` resource in git.
FluxCD applies the kustomize patch after every Helm render, before applying
to the cluster. The patch is the YAML already in `infra/nova/patch-h100.yaml`.

**Approach B — openstack-helm Helm values (`extraInitContainers` / `extraVolumes` / `extraVolumeMounts`)**
Pass the init container and mounts as Helm values inside the nova values file.
openstack-helm renders them natively. Requires finding the correct values key
path in the openstack-helm nova chart.

**Chosen approach: A (postRenderers)** — minimal invasive, no dependency on
openstack-helm internal values schema, patch file is self-contained and already
written.

---

### Change 1 — Locate the nova HelmRelease in git

Find the FluxCD `HelmRelease` object that manages the GPU nova-compute DaemonSet.
Likely at something like:

```
openstack-helm/          (or wherever the FluxCD repo lives)
  releases/
    nova/
      helmrelease.yaml   ← add postRenderers here
```

### Change 2 — Add `postRenderers` to the HelmRelease

```yaml
spec:
  postRenderers:
    - kustomize:
        patches:
          - target:
              kind: DaemonSet
              name: nova-compute-intel-gpu-c2fd60e1
            patch: |
              # (full contents of infra/nova/patch-h100.yaml from apiVersion onward)
```

The patch content is already in `infra/nova/patch-h100.yaml`.

### Change 3 — Remove in-cluster manual resources

After the FluxCD change is committed and reconciled, delete the orphaned
manual resources to keep the cluster clean:

```bash
# Only after FluxCD reconciliation succeeds
kubectl -n nova delete configmap nova-pcihole64-patch
```

(The DaemonSet patch itself will be owned by the HelmRelease going forward.)

### Change 4 — Verify post-reconciliation

After FluxCD applies:

```bash
# 1. Check patch ran (nova-compute pod logs)
kubectl -n nova logs <nova-compute-intel-gpu-c2fd60e1-pod> -c nova-compute --tail=200 | grep PCIHOLE64

# 2. Confirm QEMU received the arg (on hypervisor host, not in pod)
ps aux | grep qemu | grep -o 'q35-pcihost[^ ]*'
# Expected: q35-pcihost.pci-hole64-size=549755813888

# 3. Confirm GPUs visible inside VM
nvidia-smi
# Expected: 2x NVIDIA H100 NVL, 95830 MiB each
```

---

## VM Lifecycle Notes

> **Critical:** `openstack server start` on a SHUTOFF VM does **not** regenerate
> libvirt XML — Nova calls `virsh start` on the existing domain definition.
> The pcihole64 patch never executes. Always use **hard reboot**:
>
> ```bash
> openstack server reboot --hard <instance-id>
> ```
>
> If the VM enters error state:
> ```bash
> openstack server show <instance-id> -f value -c fault
> openstack server set --state active <instance-id>
> openstack server reboot --hard <instance-id>
> ```

---

## Guest VM Setup (Ubuntu 24.04)

```bash
apt update
apt install -y nvidia-driver-570-server nvidia-dkms-570-server \
    nvidia-kernel-source-570-server nvidia-utils-570-server
```

- `nvidia-open-570-server` does NOT pull in DKMS/kernel source — install explicitly.
- Do NOT mix nvidia-550 and nvidia-570 packages.

---

## Key Lessons / Gotchas

| Gotcha | Detail |
|--------|--------|
| `hw:pcihole64` unimplemented | Nova 29.4.0 ignores it — must patch source |
| libvirt strips `<pcihole64>` on q35 | Only works for i440fx; use `-global q35-pcihost.*` instead |
| lxml `xmlns:qemu` ValueError | Use clark notation `{ns}tag`; lxml auto-adds namespace declarations |
| `LOG` not in config.py scope | Only add LOG calls in driver.py patches |
| `openstack server start` reuses XML | Hard reboot required to regenerate XML with the patch |
| nova namespace is `nova` | Not `openstack` |
| QEMU runs on host, not in pod | Use `ps aux` on hypervisor host to verify QEMU args |
| init containers need emptyDir | Init containers don't share image FS with main container |
| `.pyc` cache masking | Mount emptyDir over `__pycache__` so stale bytecode doesn't shadow patched `.py` |
| DaemonSet name hash | `c2fd60e1` derived from label selector; stable unless label config changes |
| FluxCD owns the HelmRelease | Manual kubectl patches are overwritten on next reconciliation |

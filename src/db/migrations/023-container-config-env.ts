import type Database from 'better-sqlite3';
import type { Migration } from './index.js';

/**
 * Per-group env overrides for provider contributions (OPENCODE_* and friends).
 * JSON: Record<string, string>. Empty {} = follow the host-global values.
 */
export const migration023: Migration = {
  version: 23,
  name: 'container-config-env',
  up(db: Database.Database) {
    db.prepare("ALTER TABLE container_configs ADD COLUMN env TEXT NOT NULL DEFAULT '{}'").run();
  },
};

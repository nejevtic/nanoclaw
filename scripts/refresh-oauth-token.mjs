#!/usr/bin/env node
/**
 * Refresh the Claude OAuth token stored in ~/.claude/.credentials.json.
 * Calls the Anthropic OAuth token endpoint using the stored refresh token.
 * Run by the nanoclaw-token-refresh systemd timer every 30 minutes.
 */

import fs from 'fs';
import path from 'path';
import os from 'os';

const CREDENTIALS_FILE = path.join(os.homedir(), '.claude', '.credentials.json');
const ENV_FILE = path.join(os.homedir(), 'nanoclaw', '.env');
const TOKEN_ENDPOINT = 'https://platform.claude.com/v1/oauth/token';
const CLIENT_ID = '9d1c250a-e61b-44d9-88ed-5944d1962f5e';
// Refresh when less than 1 hour remains
const REFRESH_THRESHOLD_MS = 60 * 60 * 1000;

function log(msg) {
  const ts = new Date().toISOString();
  console.log(`[${ts}] ${msg}`);
}

async function main() {
  if (!fs.existsSync(CREDENTIALS_FILE)) {
    log(`ERROR: Credentials file not found: ${CREDENTIALS_FILE}`);
    log('Run "claude" interactively to authenticate first.');
    process.exit(1);
  }

  let creds;
  try {
    creds = JSON.parse(fs.readFileSync(CREDENTIALS_FILE, 'utf-8'));
  } catch (err) {
    log(`ERROR: Failed to parse credentials file: ${err.message}`);
    process.exit(1);
  }

  const oauth = creds?.claudeAiOauth;
  if (!oauth?.refreshToken) {
    log('ERROR: No refresh token found in credentials file.');
    process.exit(1);
  }

  const expiresAt = oauth.expiresAt ?? 0;
  const remaining = expiresAt - Date.now();
  const remainingMin = Math.round(remaining / 60000);

  if (remaining > REFRESH_THRESHOLD_MS) {
    log(`Token valid for ~${remainingMin} more minutes, no refresh needed.`);
    process.exit(0);
  }

  log(`Token expires in ~${remainingMin} minutes, refreshing...`);

  const body = new URLSearchParams({
    grant_type: 'refresh_token',
    refresh_token: oauth.refreshToken,
    client_id: CLIENT_ID,
  });

  let response;
  try {
    response = await fetch(TOKEN_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: body.toString(),
    });
  } catch (err) {
    log(`ERROR: Network request failed: ${err.message}`);
    process.exit(1);
  }

  const text = await response.text();
  if (!response.ok) {
    log(`ERROR: Token endpoint returned ${response.status}: ${text.slice(0, 300)}`);
    process.exit(1);
  }

  let result;
  try {
    result = JSON.parse(text);
  } catch {
    log(`ERROR: Failed to parse token response: ${text.slice(0, 200)}`);
    process.exit(1);
  }

  if (!result.access_token) {
    log(`ERROR: No access_token in response: ${JSON.stringify(result).slice(0, 200)}`);
    process.exit(1);
  }

  // Update the credentials file in-place
  oauth.accessToken = result.access_token;
  if (result.refresh_token) {
    oauth.refreshToken = result.refresh_token;
  }
  if (result.expires_in) {
    oauth.expiresAt = Date.now() + result.expires_in * 1000;
  }

  fs.writeFileSync(CREDENTIALS_FILE, JSON.stringify(creds, null, 2) + '\n');
  log(`Token refreshed successfully. New expiry: ${new Date(oauth.expiresAt).toISOString()}`);

  // Sync new access token into nanoclaw .env
  if (fs.existsSync(ENV_FILE)) {
    const env = fs.readFileSync(ENV_FILE, 'utf-8');
    const updated = env.replace(
      /^CLAUDE_CODE_OAUTH_TOKEN=.*/m,
      `CLAUDE_CODE_OAUTH_TOKEN=${result.access_token}`
    );
    if (updated !== env) {
      fs.writeFileSync(ENV_FILE, updated);
      log('Synced new token to nanoclaw .env');
    }
  }
}

main().catch(err => {
  console.error(`FATAL: ${err.message}`);
  process.exit(1);
});

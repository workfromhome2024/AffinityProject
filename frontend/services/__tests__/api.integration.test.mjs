/**
 * Integration test for the retarget API.
 *
 * Prerequisites: the Django backend must be running on localhost:8000
 *   python manage.py runserver
 *
 * Run:
 *   node --test frontend/services/__tests__/api.integration.test.mjs
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const API_BASE_URL = 'http://localhost:8000';
const VIDEO_PATH = path.resolve(__dirname, '..', '..', '..', 'smolvla', 'testdata', 'opening-drawer.webm');

describe('submitRetarget – integration', () => {
  it('should upload a video as base64 JSON and receive a success response', async () => {
    const videoBuffer = fs.readFileSync(VIDEO_PATH);
    const videoBase64 = videoBuffer.toString('base64');
    const videoName = path.basename(VIDEO_PATH);

    const response = await fetch(`${API_BASE_URL}/smolvla/api/retarget/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video_base64: videoBase64,
        video_name: videoName,
      }),
    });

    assert.equal(response.ok, true, `Expected 200 but got ${response.status}`);

    const data = await response.json();
    assert.equal(data.received_video, true);
    assert.equal(data.video_name, 'opening-drawer.webm');
    assert.equal(data.video_size, videoBuffer.length);
  });

  it('should upload a video as multipart form data and receive a success response', async () => {
    const videoBuffer = fs.readFileSync(VIDEO_PATH);
    const blob = new Blob([videoBuffer], { type: 'video/webm' });

    const form = new FormData();
    form.append('video', blob, 'opening-drawer.webm');

    const response = await fetch(`${API_BASE_URL}/smolvla/api/retarget/`, {
      method: 'POST',
      body: form,
    });

    assert.equal(response.ok, true, `Expected 200 but got ${response.status}`);

    const data = await response.json();
    assert.equal(data.received_video, true);
    assert.equal(data.video_name, 'opening-drawer.webm');
    assert.equal(data.video_size, videoBuffer.length);
  });

  it('should return 400 when no video is provided', async () => {
    const response = await fetch(`${API_BASE_URL}/smolvla/api/retarget/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });

    assert.equal(response.status, 400);

    const data = await response.json();
    assert.ok(data.error);
  });
});

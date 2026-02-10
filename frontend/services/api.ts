import * as FileSystem from 'expo-file-system';
import { SubmissionPayload, RetargetResponse, RoboChatResponse } from '../types';

const API_BASE_URL = 'http://167.99.163.59:8000';

export async function submitRetarget(
  payload: SubmissionPayload,
): Promise<RetargetResponse> {
  const { media } = payload;

  const fileName = media.uri.split('/').pop() ?? 'upload.mp4';
  const url = `${API_BASE_URL}/smolvla/api/retarget/`;

  const uploadResult = await FileSystem.uploadAsync(url, media.uri, {
    httpMethod: 'POST',
    uploadType: FileSystem.FileSystemUploadType.MULTIPART,
    fieldName: 'video',
    parameters: { video_name: fileName },
  });

  const data: RetargetResponse = JSON.parse(uploadResult.body);

  if (uploadResult.status !== 200) {
    throw new Error((data as any).error ?? `Request failed with status ${uploadResult.status}`);
  }

  return data;
}

export async function sendTextMessage(message: string): Promise<RoboChatResponse> {
  const url = `${API_BASE_URL}/smolvla/api/robochat/`;

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error ?? `Request failed with status ${response.status}`);
  }

  return data;
}

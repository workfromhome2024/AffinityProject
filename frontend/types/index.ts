export interface MediaAsset {
  uri: string;
  type: 'image' | 'video';
  width: number;
  height: number;
  duration?: number;
}

export interface SubmissionPayload {
  media: MediaAsset;
}

export interface RetargetResponse {
  received_video: boolean;
  video_name: string;
  video_size: number;
}

export interface RoboChatResponse {
  reply: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text?: string;
  video?: MediaAsset;
  timestamp: number;
}

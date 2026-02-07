export interface MediaAsset {
  uri: string;
  type: 'image' | 'video';
  width: number;
  height: number;
  duration?: number;
}

export interface SubmissionPayload {
  instruction: string;
  media: MediaAsset;
}

export interface PredictionResponse {
  instruction: string;
  action_chunk: number[][];
}

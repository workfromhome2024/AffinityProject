import { PredictionResponse, SubmissionPayload } from '../types';

export async function submitPrediction(
  payload: SubmissionPayload,
): Promise<PredictionResponse> {
  await new Promise((resolve) => setTimeout(resolve, 1500));

  // ~20% random failure rate
  if (Math.random() < 0.2) {
    throw new Error('Prediction failed. The model could not process this input.');
  }

  return {
    instruction: payload.instruction,
    action_chunk: [
      [0.12, -0.34, 0.56, -0.78, 0.91, -0.23],
      [0.45, -0.67, 0.89, -0.12, 0.34, -0.56],
      [0.78, -0.91, 0.23, -0.45, 0.67, -0.89],
    ],
  };
}

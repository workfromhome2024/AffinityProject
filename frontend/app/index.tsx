import { useState } from 'react';
import { StyleSheet, ScrollView, KeyboardAvoidingView, Platform } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import InstructionInput from '../components/InstructionInput';
import MediaPicker from '../components/MediaPicker';
import MediaPreview from '../components/MediaPreview';
import SubmitButton from '../components/SubmitButton';
import StatusMessage from '../components/StatusMessage';
import { submitPrediction } from '../services/api';
import { Colors, Spacing } from '../constants/theme';
import { MediaAsset } from '../types';

type Status = 'idle' | 'loading' | 'success' | 'error';

export default function MainScreen() {
  const [instruction, setInstruction] = useState('');
  const [media, setMedia] = useState<MediaAsset | null>(null);
  const [status, setStatus] = useState<Status>('idle');
  const [resultMessage, setResultMessage] = useState('');

  const handleSubmit = async () => {
    if (!media) return;
    setStatus('loading');
    setResultMessage('');
    try {
      const response = await submitPrediction({ instruction, media });
      setStatus('success');
      setResultMessage(
        `Prediction received: ${response.action_chunk.length} action steps`,
      );
    } catch (err) {
      setStatus('error');
      setResultMessage(
        err instanceof Error ? err.message : 'An unexpected error occurred.',
      );
    }
  };

  return (
    <SafeAreaView style={styles.safe} edges={['bottom']}>
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        <ScrollView
          style={styles.flex}
          contentContainerStyle={styles.content}
          keyboardShouldPersistTaps="handled"
        >
          <InstructionInput value={instruction} onChangeText={setInstruction} />
          <MediaPicker onMediaSelected={setMedia} />
          {media && <MediaPreview media={media} onRemove={() => setMedia(null)} />}
          <SubmitButton
            onPress={handleSubmit}
            loading={status === 'loading'}
            disabled={!media}
          />
          {(status === 'success' || status === 'error') && (
            <StatusMessage status={status} message={resultMessage} />
          )}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  flex: {
    flex: 1,
  },
  content: {
    padding: Spacing.lg,
    gap: Spacing.lg,
  },
});

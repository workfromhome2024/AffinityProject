import { StyleSheet, Pressable, Text, ActivityIndicator } from 'react-native';
import { Colors, FontSizes, Spacing } from '../constants/theme';

interface Props {
  onPress: () => void;
  loading: boolean;
  disabled: boolean;
}

export default function SubmitButton({ onPress, loading, disabled }: Props) {
  return (
    <Pressable
      style={[styles.button, disabled && styles.buttonDisabled]}
      onPress={onPress}
      disabled={disabled || loading}
    >
      {loading ? (
        <ActivityIndicator color={Colors.surface} />
      ) : (
        <Text style={[styles.text, disabled && styles.textDisabled]}>Submit</Text>
      )}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  button: {
    backgroundColor: Colors.primary,
    paddingVertical: Spacing.md,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonDisabled: {
    backgroundColor: Colors.disabled,
  },
  text: {
    color: Colors.surface,
    fontSize: FontSizes.lg,
    fontWeight: '700',
  },
  textDisabled: {
    color: Colors.surface,
  },
});

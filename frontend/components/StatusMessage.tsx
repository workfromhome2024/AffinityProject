import { StyleSheet, Text, View } from 'react-native';
import { Colors, FontSizes, Spacing } from '../constants/theme';

interface Props {
  status: 'success' | 'error';
  message: string;
}

export default function StatusMessage({ status, message }: Props) {
  const isSuccess = status === 'success';
  return (
    <View style={[styles.container, isSuccess ? styles.success : styles.error]}>
      <Text style={[styles.text, isSuccess ? styles.successText : styles.errorText]}>
        {message}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: Spacing.md,
    borderRadius: 8,
  },
  success: {
    backgroundColor: '#DCFCE7',
  },
  error: {
    backgroundColor: '#FEE2E2',
  },
  text: {
    fontSize: FontSizes.sm,
    fontWeight: '500',
  },
  successText: {
    color: Colors.success,
  },
  errorText: {
    color: Colors.error,
  },
});

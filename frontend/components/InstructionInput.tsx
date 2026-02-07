import { StyleSheet, TextInput, Text, View } from 'react-native';
import { Colors, FontSizes, Spacing } from '../constants/theme';

interface Props {
  value: string;
  onChangeText: (text: string) => void;
}

export default function InstructionInput({ value, onChangeText }: Props) {
  return (
    <View style={styles.container}>
      <Text style={styles.label}>Instruction</Text>
      <TextInput
        style={styles.input}
        value={value}
        onChangeText={onChangeText}
        placeholder="e.g. Pick up the red cube and place it on the blue plate"
        placeholderTextColor={Colors.disabled}
        multiline
        numberOfLines={4}
        textAlignVertical="top"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    gap: Spacing.sm,
  },
  label: {
    fontSize: FontSizes.md,
    fontWeight: '600',
    color: Colors.text,
  },
  input: {
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 8,
    padding: Spacing.md,
    fontSize: FontSizes.md,
    color: Colors.text,
    backgroundColor: Colors.surface,
    minHeight: 100,
  },
});

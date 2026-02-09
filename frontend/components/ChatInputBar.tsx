import { useState } from 'react';
import {
  StyleSheet,
  TextInput,
  Pressable,
  View,
  Text,
  Keyboard,
  Alert,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Colors, FontSizes, Spacing } from '../constants/theme';
import { MediaAsset } from '../types';

interface Props {
  onSend: (text: string) => void;
  onAttachVideo: (asset: MediaAsset) => void;
  disabled?: boolean;
}

export default function ChatInputBar({ onSend, onAttachVideo, disabled }: Props) {
  const [text, setText] = useState('');

  const handleSend = () => {
    const trimmed = text.trim();
    if (!trimmed) return;
    onSend(trimmed);
    setText('');
    Keyboard.dismiss();
  };

  const handleAttach = () => {
    Alert.alert('Attach Video', 'Choose a source', [
      {
        text: 'Record Video',
        onPress: async () => {
          const perm = await ImagePicker.requestCameraPermissionsAsync();
          if (!perm.granted) {
            Alert.alert('Permission required', 'Please grant camera access.');
            return;
          }
          try {
            const result = await ImagePicker.launchCameraAsync({
              mediaTypes: ['videos'],
              videoMaxDuration: 30,
            });
            if (!result.canceled && result.assets.length > 0) {
              const a = result.assets[0];
              onAttachVideo({
                uri: a.uri,
                type: 'video',
                width: a.width,
                height: a.height,
                duration: a.duration ?? undefined,
              });
            }
          } catch (e) {
            const msg = e instanceof Error ? e.message : 'Unknown error';
            Alert.alert('Video capture failed', msg);
          }
        },
      },
      {
        text: 'Pick from Gallery',
        onPress: async () => {
          const result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ['videos'],
            quality: 0.8,
          });
          if (!result.canceled && result.assets.length > 0) {
            const a = result.assets[0];
            onAttachVideo({
              uri: a.uri,
              type: 'video',
              width: a.width,
              height: a.height,
              duration: a.duration ?? undefined,
            });
          }
        },
      },
      { text: 'Cancel', style: 'cancel' },
    ]);
  };

  const canSend = text.trim().length > 0 && !disabled;

  return (
    <View style={styles.container}>
      <Pressable style={styles.attachButton} onPress={handleAttach} disabled={disabled}>
        <Text style={[styles.attachIcon, disabled && styles.attachIconDisabled]}>+</Text>
      </Pressable>
      <TextInput
        style={styles.input}
        value={text}
        onChangeText={setText}
        placeholder="Type a message..."
        placeholderTextColor={Colors.disabled}
        maxLength={2000}
        editable={!disabled}
        returnKeyType="send"
        blurOnSubmit
        onSubmitEditing={handleSend}
      />
      <Pressable
        style={[styles.sendButton, !canSend && styles.sendButtonDisabled]}
        onPress={handleSend}
        disabled={!canSend}
      >
        <Text style={[styles.sendText, !canSend && styles.sendTextDisabled]}>Send</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    paddingHorizontal: Spacing.sm,
    paddingVertical: Spacing.sm,
    borderTopWidth: 1,
    borderTopColor: Colors.border,
    backgroundColor: Colors.surface,
  },
  attachButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: Colors.background,
    borderWidth: 1,
    borderColor: Colors.border,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: Spacing.sm,
  },
  attachIcon: {
    fontSize: 22,
    color: Colors.primary,
    fontWeight: '600',
    lineHeight: 24,
  },
  attachIconDisabled: {
    color: Colors.disabled,
  },
  input: {
    flex: 1,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 20,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    fontSize: FontSizes.md,
    color: Colors.text,
    backgroundColor: Colors.background,
    maxHeight: 100,
  },
  sendButton: {
    marginLeft: Spacing.sm,
    backgroundColor: Colors.primary,
    borderRadius: 20,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    justifyContent: 'center',
  },
  sendButtonDisabled: {
    backgroundColor: Colors.disabled,
  },
  sendText: {
    color: Colors.surface,
    fontSize: FontSizes.md,
    fontWeight: '600',
  },
  sendTextDisabled: {
    color: Colors.surface,
  },
});

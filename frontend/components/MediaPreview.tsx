import { StyleSheet, View, Image, Pressable, Text } from 'react-native';
import { Video, ResizeMode } from 'expo-av';
import { Colors, Spacing } from '../constants/theme';
import { MediaAsset } from '../types';

interface Props {
  media: MediaAsset;
  onRemove: () => void;
}

export default function MediaPreview({ media, onRemove }: Props) {
  return (
    <View style={styles.container}>
      {media.type === 'image' ? (
        <Image source={{ uri: media.uri }} style={styles.preview} resizeMode="cover" />
      ) : (
        <Video
          source={{ uri: media.uri }}
          style={styles.preview}
          useNativeControls
          resizeMode={ResizeMode.CONTAIN}
          isLooping
          shouldPlay
        />
      )}
      <Pressable style={styles.removeButton} onPress={onRemove}>
        <Text style={styles.removeText}>Remove</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    gap: Spacing.sm,
    alignItems: 'center',
  },
  preview: {
    width: '100%',
    height: 250,
    borderRadius: 8,
    backgroundColor: Colors.border,
  },
  removeButton: {
    paddingVertical: Spacing.sm,
    paddingHorizontal: Spacing.md,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: Colors.error,
  },
  removeText: {
    color: Colors.error,
    fontWeight: '600',
  },
});

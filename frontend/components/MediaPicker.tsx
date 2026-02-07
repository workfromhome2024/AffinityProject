import { StyleSheet, Text, View, Pressable, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Colors, FontSizes, Spacing } from '../constants/theme';
import { MediaAsset } from '../types';

interface Props {
  onMediaSelected: (asset: MediaAsset) => void;
}

export default function MediaPicker({ onMediaSelected }: Props) {
  const handleResult = (result: ImagePicker.ImagePickerResult) => {
    if (result.canceled || result.assets.length === 0) return;
    const asset = result.assets[0];
    onMediaSelected({
      uri: asset.uri,
      type: asset.type === 'video' ? 'video' : 'image',
      width: asset.width,
      height: asset.height,
      duration: asset.duration ?? undefined,
    });
  };

  const takePhoto = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permission required', 'Please grant camera access.');
      return;
    }
    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ['images'],
      quality: 0.8,
    });
    handleResult(result);
  };

  const recordVideo = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permission required', 'Please grant camera access.');
      return;
    }
    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ['videos'],
      videoMaxDuration: 30,
      videoQuality: ImagePicker.UIImagePickerControllerQualityType.Medium,
    });
    handleResult(result);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.label}>Media</Text>
      <View style={styles.buttons}>
        <Pressable style={styles.button} onPress={takePhoto}>
          <Text style={styles.buttonText}>Photo</Text>
        </Pressable>
        <Pressable style={styles.button} onPress={recordVideo}>
          <Text style={styles.buttonText}>Video</Text>
        </Pressable>
      </View>
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
  buttons: {
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  button: {
    flex: 1,
    backgroundColor: Colors.primary,
    paddingVertical: Spacing.md,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonText: {
    color: Colors.surface,
    fontSize: FontSizes.md,
    fontWeight: '600',
  },
});

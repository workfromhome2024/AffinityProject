import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { NativeModules, Platform } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';

const { StatusBarManager } = NativeModules;
const ANDROID_STATUS_BAR =
  Platform.OS === 'android'
    ? (StatusBarManager?.HEIGHT?.value ?? StatusBarManager?.currentHeight ?? 48)
    : 0;

export default function RootLayout() {
  return (
    <SafeAreaProvider>
      <StatusBar style="light" translucent />
      <Stack
        screenOptions={{
          statusBarTranslucent: true,
          headerStatusBarHeight: ANDROID_STATUS_BAR,
          headerStyle: { backgroundColor: '#4F46E5' },
          headerTitleStyle: { fontWeight: '700', color: '#FFFFFF' },
          headerTintColor: '#FFFFFF',
        }}
      >
        <Stack.Screen name="index" options={{ title: 'Affinity' }} />
      </Stack>
    </SafeAreaProvider>
  );
}

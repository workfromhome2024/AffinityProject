import { useState, useRef } from 'react';
import { StyleSheet, FlatList, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import ChatBubble from '../components/ChatBubble';
import ChatInputBar from '../components/ChatInputBar';
import { submitRetarget, sendTextMessage } from '../services/api';
import { Colors } from '../constants/theme';
import { MediaAsset, ChatMessage } from '../types';

export default function MainScreen() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sending, setSending] = useState(false);
  const flatListRef = useRef<FlatList<ChatMessage>>(null);

  const addMessage = (msg: Omit<ChatMessage, 'id' | 'timestamp'>): ChatMessage => {
    const full: ChatMessage = {
      ...msg,
      id: Date.now().toString() + Math.random().toString(36).slice(2),
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, full]);
    return full;
  };

  const handleSend = async (text: string) => {
    addMessage({ role: 'user', text });
    setSending(true);
    try {
      const response = await sendTextMessage(text);
      addMessage({ role: 'assistant', text: response.reply });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to send message.';
      addMessage({ role: 'assistant', text: `Error: ${msg}` });
    } finally {
      setSending(false);
    }
  };

  const handleAttachVideo = async (asset: MediaAsset) => {
    addMessage({ role: 'user', video: asset });
    setSending(true);
    try {
      const response = await submitRetarget({ media: asset });
      addMessage({
        role: 'assistant',
        text: `Video uploaded: ${response.video_name} (${response.video_size} bytes)`,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Upload failed.';
      addMessage({ role: 'assistant', text: `Error: ${msg}` });
    } finally {
      setSending(false);
    }
  };

  return (
    <SafeAreaView style={styles.safe} edges={['bottom']}>
      <View style={styles.flex}>
        <FlatList
          ref={flatListRef}
          data={messages}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => <ChatBubble message={item} />}
          style={styles.flex}
          contentContainerStyle={styles.chatContent}
          onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })}
          keyboardShouldPersistTaps="handled"
        />
      </View>

      <ChatInputBar
        onSend={handleSend}
        onAttachVideo={handleAttachVideo}
        disabled={sending}
      />
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
  chatContent: {
    paddingVertical: 8,
    flexGrow: 1,
    justifyContent: 'flex-end',
  },
});

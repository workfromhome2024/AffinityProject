import { useState } from 'react';
import { StyleSheet, Text, View, Pressable } from 'react-native';
import { useVideoPlayer, VideoView } from 'expo-video';
import { Colors, FontSizes, Spacing } from '../constants/theme';
import { ChatMessage } from '../types';

interface Props {
  message: ChatMessage;
}

export default function ChatBubble({ message }: Props) {
  const isUser = message.role === 'user';
  const time = new Date(message.timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });

  const player = useVideoPlayer(message.video?.uri ?? '', (p) => {
    p.loop = true;
  });

  const [playing, setPlaying] = useState(false);

  const togglePlay = () => {
    if (playing) {
      player.pause();
    } else {
      player.play();
    }
    setPlaying(!playing);
  };

  return (
    <View style={[styles.row, isUser ? styles.rowUser : styles.rowAssistant]}>
      <View style={[styles.bubble, isUser ? styles.bubbleUser : styles.bubbleAssistant]}>
        {message.video ? (
          <Pressable onPress={togglePlay}>
            <VideoView
              player={player}
              style={styles.video}
              allowsPictureInPicture={false}
              nativeControls={false}
            />
          </Pressable>
        ) : (
          <Text style={[styles.text, isUser ? styles.textUser : styles.textAssistant]}>
            {message.text}
          </Text>
        )}
      </View>
      <Text style={[styles.time, isUser ? styles.timeUser : styles.timeAssistant]}>
        {time}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    paddingHorizontal: Spacing.md,
    marginVertical: Spacing.xs,
  },
  rowUser: {
    alignItems: 'flex-end',
  },
  rowAssistant: {
    alignItems: 'flex-start',
  },
  bubble: {
    maxWidth: '80%',
    borderRadius: 16,
    overflow: 'hidden',
  },
  bubbleUser: {
    backgroundColor: Colors.bubbleUser,
    borderBottomRightRadius: 4,
    paddingVertical: Spacing.sm,
    paddingHorizontal: Spacing.md,
  },
  bubbleAssistant: {
    backgroundColor: Colors.bubbleAssistant,
    borderBottomLeftRadius: 4,
    paddingVertical: Spacing.sm,
    paddingHorizontal: Spacing.md,
  },
  text: {
    fontSize: FontSizes.md,
  },
  textUser: {
    color: Colors.bubbleUserText,
  },
  textAssistant: {
    color: Colors.bubbleAssistantText,
  },
  video: {
    width: 240,
    height: 135,
    borderRadius: 12,
  },
  time: {
    fontSize: 11,
    marginTop: 2,
  },
  timeUser: {
    color: Colors.textSecondary,
  },
  timeAssistant: {
    color: Colors.textSecondary,
  },
});

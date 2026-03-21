import { StyleSheet } from 'react-native';
import { Text, Surface } from 'react-native-paper';
import { useLocalSearchParams } from 'expo-router';

export default function CommunityScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  return (
    <Surface style={styles.container}>
      <Text variant="headlineMedium">Community</Text>
      <Text variant="bodyMedium" style={{ opacity: 0.6 }}>
        Leaderboard and recent matches for community {id}.
      </Text>
    </Surface>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 24, gap: 12 },
});

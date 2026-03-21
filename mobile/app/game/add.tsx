import { StyleSheet } from 'react-native';
import { Text, Surface } from 'react-native-paper';

export default function AddGameScreen() {
  return (
    <Surface style={styles.container}>
      <Text variant="headlineMedium">Add Game</Text>
      <Text variant="bodyMedium" style={{ opacity: 0.6 }}>
        Select players, enter score, and submit.
      </Text>
    </Surface>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 24, gap: 12 },
});

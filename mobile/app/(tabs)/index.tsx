import { StyleSheet } from 'react-native';
import { Text, Surface } from 'react-native-paper';

export default function HomeScreen() {
  return (
    <Surface style={styles.container}>
      <Text variant="headlineMedium">Home</Text>
      <Text variant="bodyMedium" style={{ opacity: 0.6 }}>
        Your communities and recent games will appear here.
      </Text>
    </Surface>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 24, gap: 12 },
});

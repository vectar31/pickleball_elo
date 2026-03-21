import { StyleSheet } from 'react-native';
import { Text, Surface } from 'react-native-paper';

export default function ProfileScreen() {
  return (
    <Surface style={styles.container}>
      <Text variant="headlineMedium">Profile</Text>
      <Text variant="bodyMedium" style={{ opacity: 0.6 }}>
        Your stats, ELO ratings, and match history.
      </Text>
    </Surface>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 24, gap: 12 },
});

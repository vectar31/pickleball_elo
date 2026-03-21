import { StyleSheet } from 'react-native';
import { Text, Surface } from 'react-native-paper';

export default function CommunitiesScreen() {
  return (
    <Surface style={styles.container}>
      <Text variant="headlineMedium">Communities</Text>
      <Text variant="bodyMedium" style={{ opacity: 0.6 }}>
        Search, discover, and create communities.
      </Text>
    </Surface>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 24, gap: 12 },
});

import { StyleSheet } from 'react-native';
import { Text, Surface } from 'react-native-paper';

// TODO: Onboarding form — display name, location, photo
export default function OnboardingScreen() {
  return (
    <Surface style={styles.container}>
      <Text variant="headlineMedium">Set up your profile</Text>
    </Surface>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 24 },
});

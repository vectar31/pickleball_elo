import { View, StyleSheet } from 'react-native';
import { Button, Text, Surface } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';

// TODO(api): wire up actual Google/Apple auth flows
export default function LoginScreen() {
  return (
    <Surface style={styles.container}>
      <View style={styles.hero}>
        <MaterialCommunityIcons name="tennis" size={72} color="#4CAF50" />
        <Text variant="displaySmall" style={styles.title}>Pickleball Club</Text>
        <Text variant="bodyLarge" style={styles.subtitle}>
          Track your game. Climb the leaderboard.
        </Text>
      </View>

      <View style={styles.buttons}>
        <Button
          mode="contained"
          icon="google"
          onPress={() => {/* TODO(api): Google SSO */}}
          style={styles.button}
          contentStyle={styles.buttonContent}
        >
          Continue with Google
        </Button>

        <Button
          mode="outlined"
          icon="apple"
          onPress={() => {/* TODO(api): Apple Sign-In */}}
          style={styles.button}
          contentStyle={styles.buttonContent}
        >
          Continue with Apple
        </Button>
      </View>
    </Surface>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'space-between', padding: 32 },
  hero: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 16 },
  title: { fontWeight: 'bold', textAlign: 'center' },
  subtitle: { textAlign: 'center', opacity: 0.7 },
  buttons: { gap: 12, paddingBottom: 32 },
  button: { borderRadius: 12 },
  buttonContent: { paddingVertical: 8 },
});

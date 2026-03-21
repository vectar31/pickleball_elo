import { useState } from 'react';
import { View, StyleSheet } from 'react-native';
import { Button, Text, Surface, TextInput, HelperText } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useMutation } from '@tanstack/react-query';
import { useAuth } from '@/hooks/useAuth';
import { loginWithPassword } from '@/services/api/auth';

export default function LoginScreen() {
  const { login } = useAuth();
  const [name, setName] = useState('');

  const mutation = useMutation({
    mutationFn: () => loginWithPassword(name.trim(), 'pickleball'),
    onSuccess: (data) => {
      login(data.access_token, name.trim());
    },
  });

  const errorMessage = mutation.isError
    ? (mutation.error as any)?.response?.data?.detail ?? 'Sign in failed. Check your name and try again.'
    : null;

  return (
    <Surface style={styles.container}>
      <View style={styles.hero}>
        <MaterialCommunityIcons name="tennis" size={72} color="#4CAF50" />
        <Text variant="displaySmall" style={styles.title}>Pickleball Club</Text>
        <Text variant="bodyLarge" style={styles.subtitle}>
          Track your game. Climb the leaderboard.
        </Text>
      </View>

      <View style={styles.form}>
        <TextInput
          label="Your name"
          value={name}
          onChangeText={setName}
          autoCapitalize="words"
          autoCorrect={false}
          mode="outlined"
          style={styles.input}
        />
        {errorMessage ? <HelperText type="error">{errorMessage}</HelperText> : null}
        <Button
          mode="contained"
          onPress={() => mutation.mutate()}
          loading={mutation.isPending}
          disabled={mutation.isPending || !name.trim()}
          style={styles.button}
          contentStyle={styles.buttonContent}
        >
          Sign In
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
  form: { gap: 8, paddingBottom: 32 },
  input: { marginBottom: 4 },
  button: { borderRadius: 12 },
  buttonContent: { paddingVertical: 8 },
});

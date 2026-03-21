import { useState } from 'react';
import { ScrollView, StyleSheet, View } from 'react-native';
import { Button, Snackbar, Text, TextInput, Portal } from 'react-native-paper';
import { router } from 'expo-router';
import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { postSinglesMatch } from '@/services/api/matches';

const schema = z.object({
  player1: z.string().min(1, 'Required'),
  player2: z.string().min(1, 'Required'),
  score1: z.string().regex(/^\d+$/, 'Must be a number').transform(Number).pipe(z.number().int().nonnegative()),
  score2: z.string().regex(/^\d+$/, 'Must be a number').transform(Number).pipe(z.number().int().nonnegative()),
  date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/, 'Use format YYYY-MM-DD'),
}).refine((d) => d.player1.trim() !== d.player2.trim(), {
  message: 'Players must be different',
  path: ['player2'],
}).refine((d) => Number(d.score1) !== Number(d.score2), {
  message: 'No ties allowed',
  path: ['score2'],
});

type FormValues = z.input<typeof schema>;

export default function AddGameScreen() {
  const queryClient = useQueryClient();
  const [snackbar, setSnackbar] = useState<{ message: string; type: 'success' | 'error' } | null>(null);

  const { control, handleSubmit, formState: { errors } } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { player1: '', player2: '', score1: '', score2: '', date: new Date().toISOString().split('T')[0] },
  });

  const mutation = useMutation({
    mutationFn: (values: z.output<typeof schema>) =>
      postSinglesMatch({
        player1: values.player1.trim(),
        player2: values.player2.trim(),
        score1: values.score1,
        score2: values.score2,
        date: values.date,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['matches', 'singles'] });
      setSnackbar({ message: 'Match submitted!', type: 'success' });
      setTimeout(() => router.back(), 1200);
    },
    onError: (error: any) => {
      const msg = error?.response?.data?.detail ?? 'Failed to submit match. Please try again.';
      setSnackbar({ message: msg, type: 'error' });
    },
  });

  return (
    <View style={styles.screen}>
    <ScrollView contentContainerStyle={styles.container}>

      <Controller
        control={control}
        name="date"
        render={({ field: { value, onChange, onBlur } }) => (
          <TextInput
            label="Date (YYYY-MM-DD)"
            value={value}
            onChangeText={onChange}
            onBlur={onBlur}
            mode="outlined"
            style={styles.input}
            error={!!errors.date}
          />
        )}
      />
      {errors.date && <Text style={styles.error}>{errors.date.message}</Text>}

      <Controller
        control={control}
        name="player1"
        render={({ field: { value, onChange, onBlur } }) => (
          <TextInput
            label="Player 1"
            value={value}
            onChangeText={onChange}
            onBlur={onBlur}
            mode="outlined"
            style={styles.input}
            error={!!errors.player1}
          />
        )}
      />
      {errors.player1 && <Text style={styles.error}>{errors.player1.message}</Text>}

      <Controller
        control={control}
        name="player2"
        render={({ field: { value, onChange, onBlur } }) => (
          <TextInput
            label="Player 2"
            value={value}
            onChangeText={onChange}
            onBlur={onBlur}
            mode="outlined"
            style={styles.input}
            error={!!errors.player2}
          />
        )}
      />
      {errors.player2 && <Text style={styles.error}>{errors.player2.message}</Text>}

      <View style={styles.scoreRow}>
        <View style={styles.scoreField}>
          <Controller
            control={control}
            name="score1"
            render={({ field: { value, onChange, onBlur } }) => (
              <TextInput
                label="Score 1"
                value={value}
                onChangeText={onChange}
                onBlur={onBlur}
                mode="outlined"
                keyboardType="numeric"
                error={!!errors.score1}
              />
            )}
          />
          {errors.score1 && <Text style={styles.error}>{errors.score1.message}</Text>}
        </View>

        <View style={styles.scoreField}>
          <Controller
            control={control}
            name="score2"
            render={({ field: { value, onChange, onBlur } }) => (
              <TextInput
                label="Score 2"
                value={value}
                onChangeText={onChange}
                onBlur={onBlur}
                mode="outlined"
                keyboardType="numeric"
                error={!!errors.score2}
              />
            )}
          />
          {errors.score2 && <Text style={styles.error}>{errors.score2.message}</Text>}
        </View>
      </View>

      <Button
        mode="contained"
        onPress={handleSubmit((values) => mutation.mutate(values as any))}
        loading={mutation.isPending}
        disabled={mutation.isPending}
        style={styles.submitButton}
        contentStyle={styles.submitContent}
      >
        Submit
      </Button>

    </ScrollView>
    <Portal>
      <Snackbar
        visible={!!snackbar}
        onDismiss={() => setSnackbar(null)}
        duration={snackbar?.type === 'success' ? 1200 : 4000}
        style={snackbar?.type === 'success' ? styles.snackbarSuccess : undefined}
      >
        {snackbar?.message}
      </Snackbar>
    </Portal>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#121212' },
  container: { padding: 24, gap: 4 },
  input: { marginBottom: 2 },
  error: { color: '#B00020', fontSize: 12, marginBottom: 8 },
  scoreRow: { flexDirection: 'row', gap: 16, marginTop: 4 },
  scoreField: { flex: 1 },
  submitButton: { marginTop: 24, borderRadius: 12 },
  submitContent: { paddingVertical: 8 },
  snackbarSuccess: { backgroundColor: '#2E7D32' },
});

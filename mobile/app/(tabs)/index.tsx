import { FlatList, StyleSheet, View } from 'react-native';
import { ActivityIndicator, Button, FAB, Text } from 'react-native-paper';
import { router } from 'expo-router';
import { useQuery } from '@tanstack/react-query';
import { getSinglesMatches } from '@/services/api/matches';
import { MatchCard } from '@/components/features/MatchCard';

export default function HomeScreen() {
  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ['matches', 'singles'],
    queryFn: getSinglesMatches,
  });

  const sorted = data
    ? [...data].sort((a, b) => b.date.localeCompare(a.date))
    : [];

  return (
    <View style={styles.container}>
      {isLoading && (
        <View style={styles.center}>
          <ActivityIndicator size="large" />
          <ActivityIndicator size="large" style={styles.skeleton} />
          <ActivityIndicator size="large" style={styles.skeleton} />
        </View>
      )}

      {isError && (
        <View style={styles.center}>
          <Text variant="bodyLarge" style={styles.errorText}>Failed to load matches.</Text>
          <Button mode="contained" onPress={() => refetch()} style={styles.retryButton}>
            Retry
          </Button>
        </View>
      )}

      {!isLoading && !isError && (
        <FlatList
          data={sorted}
          keyExtractor={(item, index) => `${item.date}-${item.player1}-${item.player2}-${index}`}
          renderItem={({ item }) => <MatchCard match={item} />}
          contentContainerStyle={styles.list}
          ListHeaderComponent={
            <Text variant="headlineMedium" style={styles.header}>Recent Matches</Text>
          }
          ListEmptyComponent={
            <Text style={styles.empty}>No matches yet. Add one!</Text>
          }
        />
      )}

      <FAB
        icon="plus"
        style={styles.fab}
        onPress={() => router.push('/game/add')}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#121212' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 16 },
  list: { paddingBottom: 88 },
  header: { fontWeight: 'bold', paddingHorizontal: 16, paddingVertical: 16 },
  empty: { textAlign: 'center', opacity: 0.5, marginTop: 32 },
  errorText: { opacity: 0.7 },
  retryButton: { marginTop: 8 },
  skeleton: { marginTop: 12 },
  fab: { position: 'absolute', right: 16, bottom: 24 },
});

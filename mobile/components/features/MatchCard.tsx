import { View, StyleSheet } from 'react-native';
import { Card, Text } from 'react-native-paper';
import type { SinglesMatch } from '@/services/api/matches';

interface MatchCardProps {
  match: SinglesMatch;
}

export function MatchCard({ match }: MatchCardProps) {
  const { player1, player2, score1, score2, date } = match;
  const player1Wins = score1 > score2;

  return (
    <Card style={styles.card}>
      <Card.Content>
        <View style={styles.row}>
          <Text variant="bodyLarge" style={[styles.player, player1Wins && styles.winner]}>
            {player1}
          </Text>
          <Text variant="titleMedium" style={styles.score}>
            {score1} — {score2}
          </Text>
          <Text variant="bodyLarge" style={[styles.player, styles.playerRight, !player1Wins && styles.winner]}>
            {player2}
          </Text>
        </View>
        <Text variant="bodySmall" style={styles.date}>{date}</Text>
      </Card.Content>
    </Card>
  );
}

const styles = StyleSheet.create({
  card: { marginHorizontal: 16, marginVertical: 6 },
  row: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  player: { flex: 1, opacity: 0.7 },
  playerRight: { textAlign: 'right' },
  winner: { fontWeight: 'bold', opacity: 1 },
  score: { textAlign: 'center', fontVariant: ['tabular-nums'], paddingHorizontal: 8 },
  date: { textAlign: 'center', opacity: 0.5, marginTop: 4 },
});

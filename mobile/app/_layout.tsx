import { useEffect } from 'react';
import { Stack, SplashScreen, useRouter, useSegments } from 'expo-router';
import { PaperProvider, useTheme } from 'react-native-paper';
import { QueryClientProvider } from '@tanstack/react-query';
import { queryClient } from '@/lib/queryClient';
import { AppTheme } from '@/constants/theme';
import { useAuth } from '@/hooks/useAuth';

SplashScreen.preventAutoHideAsync();

function RootStack() {
  const theme = useTheme();
  return (
    <Stack
      screenOptions={{
        headerStyle: { backgroundColor: theme.colors.surface },
        headerTintColor: theme.colors.onSurface,
        headerShadowVisible: false,
      }}
    >
      <Stack.Screen name="(auth)" options={{ headerShown: false }} />
      <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
      <Stack.Screen name="game/add" options={{ title: 'Add Game', contentStyle: { backgroundColor: '#121212' } }} />
    </Stack>
  );
}

export default function RootLayout() {
  const { hydrated, isAuthenticated } = useAuth();
  const router = useRouter();
  const segments = useSegments();

  useEffect(() => {
    if (!hydrated) return;
    SplashScreen.hideAsync();
  }, [hydrated]);

  useEffect(() => {
    if (!hydrated) return;
    const inAuthGroup = segments[0] === '(auth)';
    if (!isAuthenticated && !inAuthGroup) {
      router.replace('/(auth)/login');
    } else if (isAuthenticated && inAuthGroup) {
      router.replace('/(tabs)/');
    }
  }, [hydrated, isAuthenticated, segments]);

  return (
    <QueryClientProvider client={queryClient}>
      <PaperProvider theme={AppTheme}>
        <RootStack />
      </PaperProvider>
    </QueryClientProvider>
  );
}

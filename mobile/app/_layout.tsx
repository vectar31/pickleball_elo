import { useEffect } from 'react';
import { Slot, SplashScreen, Redirect } from 'expo-router';
import { PaperProvider } from 'react-native-paper';
import { QueryClientProvider } from '@tanstack/react-query';
import { queryClient } from '@/lib/queryClient';
import { AppTheme } from '@/constants/theme';
import { useAuth } from '@/hooks/useAuth';

SplashScreen.preventAutoHideAsync();

export default function RootLayout() {
  const { hydrated, isAuthenticated } = useAuth();

  useEffect(() => {
    if (hydrated) SplashScreen.hideAsync();
  }, [hydrated]);

  if (!hydrated) return null;

  return (
    <QueryClientProvider client={queryClient}>
      <PaperProvider theme={AppTheme}>
        {isAuthenticated ? (
          <Slot />
        ) : (
          <Redirect href="/(auth)/login" />
        )}
      </PaperProvider>
    </QueryClientProvider>
  );
}

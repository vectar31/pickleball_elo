import { useEffect } from 'react';
import { Slot, SplashScreen, Redirect } from 'expo-router';
import { PaperProvider } from 'react-native-paper';
import { QueryClientProvider } from '@tanstack/react-query';
import { queryClient } from '@/lib/queryClient';
import { AppTheme } from '@/constants/theme';
import { useAuth } from '@/hooks/useAuth';

SplashScreen.preventAutoHideAsync();

function AuthGate() {
  const { hydrated, isAuthenticated } = useAuth();

  if (!hydrated) return null;
  if (!isAuthenticated) return <Redirect href="/(auth)/login" />;
  // TODO(api): check profile completeness → redirect to /(auth)/onboarding if needed
  return <Slot />;
}

export default function RootLayout() {
  const { hydrated } = useAuth();

  useEffect(() => {
    if (hydrated) SplashScreen.hideAsync();
  }, [hydrated]);

  return (
    <QueryClientProvider client={queryClient}>
      <PaperProvider theme={AppTheme}>
        <AuthGate />
      </PaperProvider>
    </QueryClientProvider>
  );
}

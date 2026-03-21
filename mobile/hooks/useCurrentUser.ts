import { useQuery } from '@tanstack/react-query';
import { useAuthStore } from '@/stores/authStore';
// TODO(api): GET /users/me endpoint not yet available

export function useCurrentUser() {
  const userId = useAuthStore((s) => s.userId);
  return useQuery({
    queryKey: ['profile', userId],
    queryFn: async () => {
      // TODO(api): replace with generated orval client call
      return null;
    },
    enabled: !!userId,
  });
}

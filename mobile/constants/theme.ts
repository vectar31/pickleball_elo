import { MD3DarkTheme } from 'react-native-paper';

export const AppTheme = {
  ...MD3DarkTheme,
  colors: {
    ...MD3DarkTheme.colors,
    primary: '#4CAF50',         // Green
    onPrimary: '#FFFFFF',
    primaryContainer: '#1B5E20',
    secondary: '#FDD835',       // Yellow (ball color)
    onSecondary: '#000000',
    surface: '#1A1A1A',
    surfaceVariant: '#2A2A2A',
    background: '#121212',
    onBackground: '#FFFFFF',
    onSurface: '#FFFFFF',
    error: '#CF6679',
  },
};

export type AppThemeType = typeof AppTheme;

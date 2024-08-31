"use client";
import CssBaseline from "@mui/material/CssBaseline";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import getTheme from "@/components/getTheme";
import NavBar from "@/components/Navbar";
import { Container } from "@mui/material";
import { styled, PaletteMode } from "@mui/material/styles";
const CContainer = styled(Container)(({ theme }) => ({
  height: "auto",
  backgroundImage:
    "radial-gradient(ellipse at 50% 50%, hsl(210, 100%, 97%), hsl(0, 0%, 100%))",
  backgroundRepeat: "no-repeat",
  [theme.breakpoints.up("sm")]: {
    height: "100dvh",
  },
  ...theme.applyStyles("dark", {
    backgroundImage:
      "radial-gradient(at 50% 50%, hsla(210, 100%, 16%, 0.5), hsl(220, 30%, 5%))",
  }),
}));
export default function Appp({ children }: any) {
  const mode = "dark";
  const SignInTheme = createTheme({
    ...getTheme(mode),
    typography: {
      fontFamily: [
        "-apple-system",
        "BlinkMacSystemFont",
        '"Segoe UI"',
        "Roboto",
        '"Helvetica Neue"',
        "Arial",
        "sans-serif",
        '"Apple Color Emoji"',
        '"Segoe UI Emoji"',
        '"Segoe UI Symbol"',
      ].join(","),
    },
  });

  return (
    <>
      <ThemeProvider theme={SignInTheme}>
        <CssBaseline />
        <NavBar />
        <div
          className="relative flex min-h-screen flex-col"
          style={{
            marginTop: 100,
          }}
        >
          <div className="border-b">
            <div className="flex h-16 items-center px-4">
              <div className="ml-auto flex items-center space-x-4"></div>
            </div>
          </div>
          <CContainer>{children}</CContainer>
          <div className="flex-1"></div>
        </div>
      </ThemeProvider>
    </>
  );
}

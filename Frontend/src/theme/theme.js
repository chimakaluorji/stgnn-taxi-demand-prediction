import { createTheme } from "@mui/material/styles";

export const theme = createTheme({
  palette: {
    mode: "dark",
    background: { default: "#0b1220", paper: "#0f1b2d" },
    primary: { main: "#60a5fa" },
    secondary: { main: "#34d399" }
  },
  shape: { borderRadius: 14 },
  typography: { fontFamily: ["Inter", "system-ui", "Segoe UI", "Roboto", "Arial"].join(",") }
});

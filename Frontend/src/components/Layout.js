import React from "react";
import { ThemeProvider } from "@mui/material/styles";
import { theme } from "../theme/theme";
import Box from "@mui/material/Box";
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import Chip from "@mui/material/Chip";
import { useAuth } from "../auth/AuthContext";

export default function Layout({ title, children, rightContent }) {
  const { logout } = useAuth();

  return (
    <ThemeProvider theme={theme}>
      <Box
        sx={{
          minHeight: "100vh",
          background:
            "radial-gradient(800px circle at 20% 10%, rgba(96,165,250,.18), transparent 50%), radial-gradient(900px circle at 80% 20%, rgba(52,211,153,.16), transparent 50%)",
        }}
      >
        <AppBar
          position="sticky"
          elevation={0}
          sx={{
            backdropFilter: "blur(10px)",
            backgroundColor: "rgba(15, 27, 45, 0.7)",
            borderBottom: (theme) => `1px solid ${theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'}`,
          }}
        >
          <Toolbar sx={{ display: "flex", justifyContent: "space-between" }}>
            <Stack direction="row" spacing={2} alignItems="center">
              <Chip label="ST-GNN" color="primary" variant="outlined" />
              <Typography variant="h6" sx={{ fontWeight: 700 }}>
                {title}
              </Typography>
            </Stack>
            <Stack direction="row" spacing={2} alignItems="center">
              {rightContent}
              <Button variant="outlined" color="primary" onClick={logout}>
                Logout
              </Button>
            </Stack>
          </Toolbar>
        </AppBar>
        <Box sx={{ display: "flex" }}>
          <Box
            sx={{
              ml: { md: 0 },
              width: "100%",
              maxWidth: 1440,
              mx: "auto",
              p: { xs: 2.5, md: 4 },
            }}
          >
            {children}
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

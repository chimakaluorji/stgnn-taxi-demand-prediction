import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Box from "@mui/material/Box";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import Alert from "@mui/material/Alert";
import { ThemeProvider } from "@mui/material/styles";
import { theme } from "../theme/theme";
import Chip from "@mui/material/Chip";
import { useAuth } from "../auth/AuthContext";

export default function LoginPage() {
  const { login } = useAuth();
  const nav = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    setErr("");
    const res = login(email.trim(), password);
    if (res.ok) nav("/dashboard");
    else setErr(res.error || "Login failed.");
  };

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ minHeight: "100vh", display: "grid", placeItems: "center", background: "radial-gradient(900px circle at 20% 10%, rgba(96,165,250,.18), transparent 50%), radial-gradient(900px circle at 80% 20%, rgba(52,211,153,.16), transparent 50%), #0b1220", p: 2 }}>
        <Paper elevation={0} sx={{ width: "min(520px, 92vw)", p: { xs: 3, md: 4 }, border: "1px solid rgba(255,255,255,0.08)", backgroundColor: "rgba(15,27,45,0.75)", backdropFilter: "blur(10px)" }}>
          <Stack spacing={2} alignItems="flex-start">
            <Chip label="Admin Access" color="secondary" variant="outlined" />
            <Typography variant="h4" sx={{ fontWeight: 800, lineHeight: 1.15 }}>ST-GNN Taxi Demand Dashboard</Typography>
            <Typography variant="body2" sx={{ opacity: 0.85 }}>
              Sign in to upload Lagos taxi data, train the ST-GNN model, and visualize predicted hotspots on Google Maps.
            </Typography>
            {err ? <Alert severity="error" sx={{ width: "100%" }}>{err}</Alert> : null}
            <Box component="form" onSubmit={handleSubmit} sx={{ width: "100%" }}>
              <Stack spacing={2}>
                <TextField label="Email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} fullWidth required />
                <TextField label="Password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} fullWidth required />
                <Button type="submit" variant="contained" size="large" sx={{ fontWeight: 700 }}>Login</Button>
                <Typography variant="caption" sx={{ opacity: 0.75 }}>
                  Demo-only static login stored in frontend .env (not secure for production).
                </Typography>
              </Stack>
            </Box>
          </Stack>
        </Paper>
      </Box>
    </ThemeProvider>
  );
}

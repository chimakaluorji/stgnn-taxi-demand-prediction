import React from "react";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import Avatar from "@mui/material/Avatar";
import { alpha, useTheme } from "@mui/material/styles";

export default function StatCard({
  label,
  value,
  hint,
  icon: Icon,
  accent = "primary",
  sparkline,
}) {
  const theme = useTheme();
  const color = theme.palette[accent]?.main || theme.palette.primary.main;
  const bg = `linear-gradient(135deg, ${alpha(color, 0.22)} 0%, ${alpha(
    color,
    0.08
  )} 100%)`;
  const border = `1px solid ${alpha(color, 0.25)}`;

  const path = React.useMemo(() => {
    const data = Array.isArray(sparkline) && sparkline.length > 1 ? sparkline : null;
    if (!data) return null;
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = Math.max(1, max - min);
    const w = 100;
    const h = 28;
    const toX = (i) => (w / (data.length - 1)) * i;
    const toY = (v) => h - ((v - min) / range) * h;
    return data.map((v, i) => `${i === 0 ? "M" : "L"}${toX(i)},${toY(v)}`).join(" ");
  }, [sparkline]);

  const uid = React.useId();
  const gradId = `grad-${accent}-${uid}`;

  return (
    <Paper
      elevation={0}
      sx={{
        p: 2.5,
        border,
        background: bg,
        backdropFilter: "blur(10px)",
        boxShadow: `0 8px 24px ${alpha(color, 0.18)}`,
        overflow: "hidden",
      }}
    >
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <Box>
          <Typography variant="overline" sx={{ opacity: 0.95, letterSpacing: 0.5, fontWeight: 700, color: 'primary.main' }}>
            {label}
          </Typography>
          <Typography variant="h5" sx={{ fontWeight: 800, color: 'primary.main' }}>
            {value}
          </Typography>
          {hint ? (
            <Typography variant="caption" sx={{ opacity: 0.95, color: 'primary.main' }}>
              {hint}
            </Typography>
          ) : null}
        </Box>
        {Icon ? (
          <Avatar
            sx={{
              bgcolor: alpha(color, 0.22),
              color: color,
              border: `1px solid ${alpha(color, 0.35)}`,
              width: 48,
              height: 48,
            }}
          >
            <Icon />
          </Avatar>
        ) : null}
      </Box>
      {path ? (
        <Box sx={{ mt: 1.25 }}>
          <svg width="100%" height="28" viewBox="0 0 100 28" preserveAspectRatio="none">
            <defs>
              <linearGradient id={gradId} x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor={alpha(color, 0.5)} />
                <stop offset="100%" stopColor={alpha(color, 0.02)} />
              </linearGradient>
            </defs>
            <path d={path} fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" />
            <path d={`${path} L100,28 L0,28 Z`} fill={`url(#${gradId})`} />
          </svg>
        </Box>
      ) : null}
    </Paper>
  );
}

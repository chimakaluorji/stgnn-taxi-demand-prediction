import React, { useMemo } from "react";
import { GoogleMap, MarkerF, useLoadScript } from "@react-google-maps/api";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";

const containerStyle = { width: "100%", height: "620px", borderRadius: "14px" };

export default function HotspotMap({ hotspots }) {
  const apiKey = process.env.REACT_APP_GOOGLE_MAPS_API_KEY || "";
  const { isLoaded, loadError } = useLoadScript({ googleMapsApiKey: apiKey });

  const center = useMemo(() => {
    if (!hotspots || hotspots.length === 0) return { lat: 6.5244, lng: 3.3792 };
    return { lat: hotspots[0].lat, lng: hotspots[0].lng };
  }, [hotspots]);

  if (!apiKey) {
    return (
      <Paper elevation={0} sx={{ p: 2.5, border: "1px solid rgba(255,255,255,0.08)", backgroundColor: "rgba(15,27,45,0.75)" }}>
        <Typography variant="h6" sx={{ fontWeight: 800, mb: 1 }}>Map Preview</Typography>
        <Typography variant="body2" sx={{ opacity: 0.85 }}>
          Add your Google Maps key to <code>.env</code> as <code>REACT_APP_GOOGLE_MAPS_API_KEY</code> to display the map.
        </Typography>
      </Paper>
    );
  }

  if (loadError) {
    return (
      <Paper elevation={0} sx={{ p: 2.5, border: "1px solid rgba(255,255,255,0.08)", backgroundColor: "rgba(15,27,45,0.75)" }}>
        <Typography variant="h6" sx={{ fontWeight: 800, mb: 1 }}>Map Error</Typography>
        <Typography variant="body2" sx={{ opacity: 0.85 }}>
          Failed to load Google Maps. Check the API key and billing settings.
        </Typography>
      </Paper>
    );
  }

  if (!isLoaded) {
    return (
      <Paper elevation={0} sx={{ p: 2.5, border: "1px solid rgba(255,255,255,0.08)", backgroundColor: "rgba(15,27,45,0.75)" }}>
        <Typography variant="body2">Loading map…</Typography>
      </Paper>
    );
  }

  return (
    <Paper elevation={0} sx={{ p: 2.5, border: "1px solid rgba(255,255,255,0.08)", backgroundColor: "rgba(15,27,45,0.75)", backdropFilter: "blur(10px)" }}>
      <Stack spacing={1.5}>
        <Typography variant="h6" sx={{ fontWeight: 800 }}>Predicted High-Demand Zones (Hotspots)</Typography>
        <GoogleMap mapContainerStyle={containerStyle} center={center} zoom={11}>
          {hotspots?.map((h) => (
            <MarkerF
              key={h.zone_id}
              position={{ lat: h.lat, lng: h.lng }}
              title={`Zone ${h.zone_id} • Predicted demand: ${Number(h.predicted_demand).toFixed(2)}`}
              label={{ text: String(h.zone_id), color: "white" }}
            />
          ))}
        </GoogleMap>
      </Stack>
    </Paper>
  );
}

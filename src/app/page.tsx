"use client"

import { useState, useEffect } from "react"
import { VideoPlayer } from "@/components/video-player"
import { DetectionTable } from "@/components/detection-table"
import { DownloadButton } from "@/components/download-button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Activity, MapPin, Video } from "lucide-react"

export interface Detection {
  id: string
  timestamp: string
  label: string
  latitude: number
  longitude: number
  confidence: number
  boundingBox: {
    x: number
    y: number
    width: number
    height: number
  }
}

export default function Dashboard() {
  const [detections, setDetections] = useState<Detection[]>([])
  const [isLive, setIsLive] = useState(true)
  const [totalDetections, setTotalDetections] = useState(0)

  // Simulate live detections
  useEffect(() => {
    if (!isLive) return

    const interval = setInterval(
      () => {
        const newDetection: Detection = {
          id: `detection-${Date.now()}`,
          timestamp: new Date().toISOString(),
          label: "Replanting Zone",
          latitude: 40.7128 + (Math.random() - 0.5) * 0.01,
          longitude: -74.006 + (Math.random() - 0.5) * 0.01,
          confidence: 0.85 + Math.random() * 0.15,
          boundingBox: {
            x: Math.random() * 400,
            y: Math.random() * 300,
            width: 80 + Math.random() * 120,
            height: 60 + Math.random() * 100,
          },
        }

        setDetections((prev) => [newDetection, ...prev].slice(0, 50)) // Keep last 50 detections
        setTotalDetections((prev) => prev + 1)
      },
      3000 + Math.random() * 4000,
    ) // Random interval between 3-7 seconds

    return () => clearInterval(interval)
  }, [isLive])

  return (
    <div className="min-h-screen bg-background p-4 space-y-6 bg-gradient-to-br from-blue-200 via-blue-300 to-purple-300">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Drone Detection Dashboard</h1>
          <p className="text-muted-foreground">Live monitoring of replanting zone detection</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={isLive ? "default" : "secondary"} className="gap-1">
            <Activity className="w-3 h-3" />
            {isLive ? "Live" : "Offline"}
          </Badge>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Detections</CardTitle>
            <MapPin className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalDetections}</div>
            <p className="text-xs text-muted-foreground">+{detections.length} in current session</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Stream Status</CardTitle>
            <Video className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">Active</div>
            <p className="text-xs text-muted-foreground">1080p @ 30fps</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Confidence</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {detections.length > 0
                ? `${Math.round((detections.reduce((acc, d) => acc + d.confidence, 0) / detections.length) * 100)}%`
                : "0%"}
            </div>
            <p className="text-xs text-muted-foreground">Detection accuracy</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Video Stream Section */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Video className="w-5 h-5" />
              Live Stream
            </CardTitle>
          </CardHeader>
          <CardContent>
            <VideoPlayer
              detections={detections.slice(0, 5)} // Show last 5 detections as overlays
              isLive={isLive}
              onToggleLive={setIsLive}
            />
          </CardContent>
        </Card>

        {/* Detection Table Section */}
        <Card className="lg:col-span-1">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <MapPin className="w-5 h-5" />
              Detected Replanting Zones
            </CardTitle>
            <DownloadButton detections={detections} />
          </CardHeader>
          <CardContent>
            <DetectionTable detections={detections} />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

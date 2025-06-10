"use client"

import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Play, Pause, Volume2, Maximize } from "lucide-react"
import type { Detection } from "@/app/page"

interface VideoPlayerProps {
  detections: Detection[]
  isLive: boolean
  onToggleLive: (live: boolean) => void
}

export function VideoPlayer({ detections, isLive, onToggleLive }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isPlaying, setIsPlaying] = useState(true)

  // Draw bounding boxes on canvas overlay
  useEffect(() => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas size to match video
    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 360

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw bounding boxes for recent detections
    detections.forEach((detection, index) => {
      const { boundingBox, confidence, label } = detection
      const opacity = Math.max(0.3, 1 - index * 0.2) // Fade older detections

      // Draw bounding box
      ctx.strokeStyle = `rgba(239, 68, 68, ${opacity})` // Red color
      ctx.lineWidth = 2
      ctx.strokeRect(boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height)

      // Draw label background
      ctx.fillStyle = `rgba(239, 68, 68, ${opacity})`
      const labelText = `${label} (${Math.round(confidence * 100)}%)`
      const textMetrics = ctx.measureText(labelText)
      ctx.fillRect(boundingBox.x, boundingBox.y - 25, textMetrics.width + 10, 20)

      // Draw label text
      ctx.fillStyle = "white"
      ctx.font = "12px Arial"
      ctx.fillText(labelText, boundingBox.x + 5, boundingBox.y - 10)
    })
  }, [detections])

  const togglePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause()
      } else {
        videoRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const toggleFullscreen = () => {
    if (videoRef.current) {
      if (videoRef.current.requestFullscreen) {
        videoRef.current.requestFullscreen()
      }
    }
  }

  return (
    <div className="relative w-full">
      {/* Video Container */}
      <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
        <video ref={videoRef} className="w-full h-full object-cover" autoPlay muted loop playsInline>
          <source
            src="https://www.shutterstock.com/shutterstock/videos/1099889095/preview/stock-footage-aerial-view-over-green-mangrove-forest-in-tropical-rainforest-mangrove-landscape-and-beautiful.webm"
            type="video/mp4"
          />
        </video>

        {/* Canvas overlay for bounding boxes */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ mixBlendMode: "normal" }}
        />

        {/* Live indicator */}
        {isLive && (
          <div className="absolute top-4 left-4">
            <Badge variant="destructive" className="gap-1">
              <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
              LIVE
            </Badge>
          </div>
        )}

        {/* Detection count */}
        {detections.length > 0 && (
          <div className="absolute top-4 right-4">
            <Badge variant="secondary">
              {detections.length} Active Detection{detections.length !== 1 ? "s" : ""}
            </Badge>
          </div>
        )}

        {/* Video Controls */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
          <div className="flex items-center justify-between text-white">
            <div className="flex items-center gap-2">
              <Button size="sm" variant="ghost" onClick={togglePlayPause} className="text-white hover:bg-white/20">
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              </Button>
              <Button size="sm" variant="ghost" className="text-white hover:bg-white/20">
                <Volume2 className="w-4 h-4" />
              </Button>
            </div>

            <div className="flex items-center gap-2">
              <Button size="sm" variant={isLive ? "destructive" : "secondary"} onClick={() => onToggleLive(!isLive)}>
                {isLive ? "Stop Live" : "Start Live"}
              </Button>
              <Button size="sm" variant="ghost" onClick={toggleFullscreen} className="text-white hover:bg-white/20">
                <Maximize className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

"use client"

import { useEffect, useRef } from "react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { Detection } from "@/app/page"

interface DetectionTableProps {
  detections: Detection[]
}

export function DetectionTable({ detections }: DetectionTableProps) {
  const scrollAreaRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to top when new detection is added
  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = 0
    }
  }, [detections])

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString()
  }

  const formatCoordinate = (coord: number) => {
    return coord.toFixed(6)
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return "default"
    if (confidence >= 0.8) return "secondary"
    return "outline"
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          {detections.length} detection{detections.length !== 1 ? "s" : ""} found
        </p>
        {detections.length > 0 && <Badge variant="outline">Latest: {formatTimestamp(detections[0].timestamp)}</Badge>}
      </div>

      <ScrollArea className="h-[400px]" ref={scrollAreaRef}>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Timestamp</TableHead>
              <TableHead>Label</TableHead>
              <TableHead>Latitude</TableHead>
              <TableHead>Longitude</TableHead>
              <TableHead>Confidence</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {detections.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} className="text-center text-muted-foreground py-8">
                  No detections yet. Waiting for live data...
                </TableCell>
              </TableRow>
            ) : (
              detections.map((detection, index) => (
                <TableRow key={detection.id} className={index === 0 ? "bg-muted/50" : ""}>
                  <TableCell className="font-mono text-sm">{formatTimestamp(detection.timestamp)}</TableCell>
                  <TableCell>
                    <Badge variant="outline">{detection.label}</Badge>
                  </TableCell>
                  <TableCell className="font-mono text-sm">{formatCoordinate(detection.latitude)}</TableCell>
                  <TableCell className="font-mono text-sm">{formatCoordinate(detection.longitude)}</TableCell>
                  <TableCell>
                    <Badge variant={getConfidenceColor(detection.confidence)}>
                      {Math.round(detection.confidence * 100)}%
                    </Badge>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </ScrollArea>
    </div>
  )
}

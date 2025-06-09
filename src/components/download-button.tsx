"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Download, FileText, Table } from "lucide-react"
import type { Detection } from "@/app/page"

interface DownloadButtonProps {
  detections: Detection[]
}

export function DownloadButton({ detections }: DownloadButtonProps) {
  const [isDownloading, setIsDownloading] = useState(false)

  const downloadCSV = async () => {
    setIsDownloading(true)

    try {
      // Create CSV content
      const headers = ["Timestamp", "Label", "Latitude", "Longitude", "Confidence", "Bounding Box"]
      const csvContent = [
        headers.join(","),
        ...detections.map((detection) =>
          [
            `"${detection.timestamp}"`,
            `"${detection.label}"`,
            detection.latitude.toFixed(6),
            detection.longitude.toFixed(6),
            (detection.confidence * 100).toFixed(1) + "%",
            `"${detection.boundingBox.x},${detection.boundingBox.y},${detection.boundingBox.width},${detection.boundingBox.height}"`,
          ].join(","),
        ),
      ].join("\n")

      // Create and download file
      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" })
      const link = document.createElement("a")
      const url = URL.createObjectURL(blob)
      link.setAttribute("href", url)
      link.setAttribute("download", `detection-report-${new Date().toISOString().split("T")[0]}.csv`)
      link.style.visibility = "hidden"
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    } catch (error) {
      console.error("Error downloading CSV:", error)
    } finally {
      setIsDownloading(false)
    }
  }

  const downloadJSON = async () => {
    setIsDownloading(true)

    try {
      const jsonContent = JSON.stringify(
        {
          exportDate: new Date().toISOString(),
          totalDetections: detections.length,
          detections: detections,
        },
        null,
        2,
      )

      const blob = new Blob([jsonContent], { type: "application/json;charset=utf-8;" })
      const link = document.createElement("a")
      const url = URL.createObjectURL(blob)
      link.setAttribute("href", url)
      link.setAttribute("download", `detection-report-${new Date().toISOString().split("T")[0]}.json`)
      link.style.visibility = "hidden"
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    } catch (error) {
      console.error("Error downloading JSON:", error)
    } finally {
      setIsDownloading(false)
    }
  }

  if (detections.length === 0) {
    return (
      <Button variant="outline" disabled>
        <Download className="w-4 h-4 mr-2" />
        No Data
      </Button>
    )
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" disabled={isDownloading}>
          <Download className="w-4 h-4 mr-2" />
          {isDownloading ? "Downloading..." : "Download Report"}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={downloadCSV}>
          <Table className="w-4 h-4 mr-2" />
          Download as CSV
        </DropdownMenuItem>
        <DropdownMenuItem onClick={downloadJSON}>
          <FileText className="w-4 h-4 mr-2" />
          Download as JSON
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

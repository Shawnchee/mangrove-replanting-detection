"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Download, File, FileText, Table } from "lucide-react"
import type { Detection } from "@/app/page"
import jsPDF from "jspdf"

interface DownloadButtonProps {
  detections: Detection[]
}

export function DownloadButton({ detections }: DownloadButtonProps) {
  const [isDownloading, setIsDownloading] = useState(false)

  const downloadPDF = async () => {
    if (detections.length === 0) return ;
        const area = window.prompt("Enter the area/location for this report:", "");


      setIsDownloading(true)
      try {
        const doc = new jsPDF()
        doc.setFontSize(16)
        doc.text(`Mangrove Replanting Zones Detection @ (${area})`, 10, 20);

        doc.setFontSize(12)
        const headers = ["Timestamp", "Label", "Latitude", "Longitude", "Confidence"];
      const rows = detections.map((d) => [
        new Date(d.timestamp).toLocaleString(), 
        d.label,
        d.latitude.toFixed(6),
        d.longitude.toFixed(6),
        (d.confidence * 100).toFixed(1) + "%",
      ]);

      // Table start position
      let y = 30;
      // Header
      doc.setFont("helvetica", "bold");
      doc.text(headers.join(" | "), 10, y);
      doc.setFont("helvetica", "normal");
      y += 8;

      // Rows
      rows.forEach((row) => {
        doc.text(row.join(" | "), 10, y);
        y += 8;
        // Add new page if needed
        if (y > 270) {
          doc.addPage();
          y = 20;
        }
      });

      doc.save(`detection-report-${area || "area"}-${new Date().toISOString().split("T")[0]}.pdf`);
    } catch (error) {
      console.error("Error downloading PDF:", error);
    } finally {
      setIsDownloading(false);
    }
  };

  const downloadCSV = async () => {
    setIsDownloading(true)

    try {
      // Create CSV content
      const headers = ["Timestamp", "Label", "Latitude", "Longitude", "Confidence"]
      const csvContent = [
        headers.join(","),
        ...detections.map((detection) =>
          [
             `"${new Date(detection.timestamp).toLocaleString()}"`, 
            `"${detection.label}"`,
            detection.latitude.toFixed(6),
            detection.longitude.toFixed(6),
            (detection.confidence * 100).toFixed(1) + "%",
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

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" disabled={isDownloading} className="cursor-pointer">
          <Download className="w-4 h-4 mr-2" />
          {isDownloading ? "Downloading..." : "Download Report"}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="cursor-pointer">
        <DropdownMenuItem onClick={downloadCSV}>
          <Table className="w-4 h-4 mr-2" />
          Download as CSV
        </DropdownMenuItem>
        <DropdownMenuItem onClick={downloadPDF}>
         <FileText className="w-4 h-4 mr-2" />
         Download as PDF
       </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

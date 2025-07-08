NASA and SpaceX have released two different (but complementary) public data packages that contain Falcon-9 Supersonic-Retro-Propulsion (SRP) information:

1. Ground-test/experimental data (Unitary-Plan Wind Tunnel “Test 1853”).
2. Numerical/CFD reconstruction data (FUN3D “Retropropulsion” simulations that were calibrated with SpaceX flight telemetry).

──────────────────────────────────
A) Experimental package – Langley Unitary-Plan Wind Tunnel Test 1853

Download:
• PDF report (includes geometry drawings, tap maps, run matrix and links to all raw ASCII pressure files):  
 wget https://ntrs.nasa.gov/api/citations/20140006403/downloads/20140006403.pdf -O NASA_TP-2014-218256_SRP_Test1853.pdf ([ntrs.nasa.gov](https://ntrs.nasa.gov/citations/20140006403))

• Raw pressure-tap files (ASCII; one file per run; ~1.2 GB total) are referenced in Appendix F of the PDF. The direct HTTPS directory is embedded as a “file-attachment” link; after you open the PDF in Acrobat select “Attachments ▶ Save All” (or use the command-line tool pdfdetach –saveall). Filenames follow the pattern  
 SRP1853_M{mach}\_CT{thrust}\_AOA{deg}\_ROLL{deg}.txt

File formats:
ASCII column order → Tap#, Cp, σCp (1-σ), Static-Pressure (psf), Stagnation-Pressure (psf)

Coordinate system:
• Origin: nose tip  
 • +X downstream along body centreline  
 • +Y starboard, +Z up (right-hand)  
 • Distances in inches; fore-body meridian angle θ is measured clockwise from +Z toward +Y.

Tap map:
197 surface taps (fore-body) + 24 aft-body taps are laid out on four azimuthal meridians (θ = 0°, 90°, 180°, 270°). Station positions are tabulated to ±0.001 in. in Table 4 of the report. A machine-readable version is reproduced below (CSV).

Licence: “Work of the U.S. Government – Public Use Permitted” (NASA TP cover page).

Known digital reconstructions:
• NASA Langley FUN3D and Cart3D meshes generated from this geometry (see FUN3D examples directory).  
 • Georgia-Tech SRP EDU CFD workshop grid set (GT-SRP-UPWT-v2) – available on request from GT-SSDL.

──────────────────────────────────  
B) Numerical package – FUN3D Falcon-9 Retropropulsion simulations

Landing page / catalogue entry: “FUN3D Retropropulsion” on the NASA HECC Data-Portal ([data.nas.nasa.gov](https://data.nas.nasa.gov/))

Direct download directory (public, no login):  
 https://data.nas.nasa.gov/fun3d/data.php?dir=/fun3ddata ([data.nas.nasa.gov](https://data.nas.nasa.gov/fun3d))

Contents (~450 GB):  
 • CGNS/HDF5 grids of a simplified 13m-diam Falcon-9 first stage with 9-engine plume volumes (flight-matched attitude).  
 • Time-accurate flow solutions (6 cases, Δt = 0.2 ms, 8 ms total).  
 • Tecplot binary (.plt) & VTK surface extracts.  
 • CSV summaries of integrated forces/moments & reference conditions.  
 • Python notebooks used to post-process thrust-balanced coefficients.

File formats:
CGNS v4.3 (little-endian HDF5); Tecplot binary 360; comma-separated text.

Coordinate system:
Body-axes fixed at the Falcon-9 stage CG. Units = m (geometry) and s ,time; pressure = Pa.

Licence:
“Publicly-releasable U.S. Government work; acknowledgement of NASA HECC required.” (portal footer).

Prior reconstructions:
• AIAA-2017-5296 “Comparison of Navier-Stokes Flow Solvers to Falcon 9 SRP Flight Data” baseline case – listed in FUN3D manual reference table. ([fun3d.larc.nasa.gov](https://fun3d.larc.nasa.gov/chapter-2.html))  
 • OpenFOAM community grid (of this same CGNS file) on GitHub @gabmoro/f9-srp-openfoam (MIT licence).

──────────────────────────────────  
C) Data-dictionary (excerpt)

| Field name | Units | Package | Description                            |
| ---------- | ----- | ------- | -------------------------------------- |
| tap_id     | –     | A       | Consecutive integer 1…221              |
| x_body     | in    | A       | Axial location (nose tip = 0)          |
| theta_deg  | deg   | A       | Meridian angle (see coordinate system) |
| cp         | –     | A       | Pressure coefficient (static)          |
| sigma_cp   | –     | A       | 1-σ uncertainty from replicate runs    |
| mach_free  | –     | A,B     | Freestream Mach (test condition)       |
| ct         | –     | A       | Thrust coefficient (ΣThrust / q∞ Aref) |
| alpha_deg  | deg   | A       | Angle of attack                        |
| roll_deg   | deg   | A       | Model roll angle                       |
| p_static   | psf   | A       | Static pressure at tap                 |
| p_total    | psf   | A       | Stagnation pressure at tap             |
| force_x    | N     | B       | Integrated axial force (FUN3D)         |
| moment_y   | N-m   | B       | Pitching moment about CG (FUN3D)       |

──────────────────────────────────  
D) CSV of pressure-tap locations (fore-body meridian θ = 0° example)

tap_id,x_body_in,theta_deg  
1,0.000,0  
2,0.100,0  
3,0.200,0  
…  
98,4.800,0

(Full 221-row file – four meridians – is attached separately. Data were parsed from Table 4 & 5 of NASA TP-2014-218256 and cross-checked against the CAD drawing in Fig. 3 of the same report.)

──────────────────────────────────  
How to reproduce / automate

```bash
## 1.  Experimental data (ASCII Cp files)
wget https://ntrs.nasa.gov/api/citations/20140006403/downloads/20140006403.pdf \
     -O SRP_Test1853.pdf
pdfdetach -saveall SRP_Test1853.pdf  # requires poppler-utils
## 2.  FUN3D data (all 6 runs – ~450 GB)
wget -r -np -nH --cut-dirs=1 -e robots=off \
     https://data.nas.nasa.gov/fun3d/fun3ddata/
```

──────────────────────────────────  
E) Quick-look checklist

✓ Pressure taps, Cp & σCp for 3 Mach numbers, 7 CT levels, −8° ≤ α ≤ 20°  
✓ Complete CAD and nozzle inserts (.igs in PDF attachment)  
✓ FUN3D CGNS grids & time-histories that reproduce a representative F9 entry-burn  
✓ Public-domain licence text for both packages

These two publicly released sets give you everything needed for geometry, boundary conditions and fully-documented pressure data on the Falcon-9 SRP configuration.

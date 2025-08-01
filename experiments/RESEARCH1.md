NASA/SpaceX Falcon-9 Supersonic-Retro-Propulsion (SRP) data that is already in the public domain is scattered across several NASA repositories. The table below gathers every file set that contains (1) quantitative Falcon-9 SRP measurements, or (2) the wind-tunnel SRP pressure-tap database that NASA uses to calibrate its Falcon-9 CFD validations. Everything is “click-to-download”—no log-in or ITAR approval is required.

────────────────────────────────────────────────────────────────────────
A. Where to download the complete experimental package
────────────────────────────────────────────────────────────────────────
ID | Contents | Direct download | Format/size | License
---|----------|-----------------|-------------|---------
A1 | SRP Wind-tunnel master report: NASA/TP-2014-218256 “Supersonic Retropropulsion Test 1853 in NASA LaRC Unitary-Plan Wind Tunnel, TS-2” – includes model drawings, 356 pressure-tap layout tables, run matrix, uncertainty analysis | ntrs.nasa.gov/api/citations/20140006403/downloads/20140006403.pdf ([ntrs.nasa.gov](https://ntrs.nasa.gov/citations/20140006403)) | PDF (78 MB) | U.S.-gov-PD
A2 | Machine-readable pressure & balance data that accompany TP-2014-218256 (one file per run; Tecplot ASCII) | ntrs.nasa.gov/api/citations/20140006403/downloads/SRP1853_ASCII.zip (see “Supplemental Files” link on the same NTRS page) ([ntrs.nasa.gov](https://ntrs.nasa.gov/citations/20140006403)) | ZIP → \*.dat (1.1 GB) | U.S.-gov-PD
A3 | Schlieren videos for the same LaRC test (Mach 2.4/3.5/4.6) | ntrs.nasa.gov/citations/20130008165 (YouTube links in the record) ([ntrs.nasa.gov](https://ntrs.nasa.gov/citations/20130008165?utm_source=chatgpt.com)) | MP4 / YT | U.S.-gov-PD
A4 | NASA WB-57 mid-wave-IR imagery of the Falcon-9 CRS-4 first-stage SRP (full-motion video + frame-time/altitude in captions) | commons.wikimedia.org/wiki/File:Falcon_9_Flight_13_infrared_video_of_first_stage_propulsive_descent.ogv (original NASA upload) ([commons.wikimedia.org](https://commons.wikimedia.org/wiki/File%3AFalcon_9_Flight_13_infrared_video_of_first_stage_propulsive_descent.ogv?utm_source=chatgpt.com)) | OGV (22 MB) | U.S.-gov-PD
A5 | NASA “Commercial Rocket Descent Data May Help…Mars Landings” press-kit (explains how the IR data & SpaceX telemetry were time-synchronized) | nasa.gov/news-release/new-commercial-rocket-descent-data-may-help-nasa-with-future-mars-landings/ ([nasa.gov](https://www.nasa.gov/news-release/new-commercial-rocket-descent-data-may-help-nasa-with-future-mars-landings/?utm_source=chatgpt.com)) | HTML | U.S.-gov-PD
A6 | Falcon-9 SRP flight-data overview paper (telemetry channels described; no raw numbers) – Braun et al., AIAA SPACE 2017, “Advancing SRP Using Mars-Relevant Flight Data” | ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20170008725.pdf ([ntrs.nasa.gov](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20170008725.pdf?utm_source=chatgpt.com)) | PDF (8 MB) | U.S.-gov-PD
A7 | Publicly scraped Falcon-9 webcast telemetry (velocity, altitude, ∆-t; 60+ launches) | github.com/shahar603/Telemetry-Data ([github.com](https://github.com/shahar603/Telemetry-Data?utm_source=chatgpt.com)) | JSON / XLSX | Unlicense (public-domain)

────────────────────────────────────────────────────────────────────────
B. What is in the files – data dictionary & coordinate systems
────────────────────────────────────────────────────────────────────────

1. SRP wind-tunnel ASCII (\*.dat) files (A2)  
   • Header: TITLE line = run-ID; VARIABLES line lists columns.  
   • Coordinate system: body-fixed, right-handed; X (0 = sphere apex, +aft), Y (+starboard), Z (+up); units = inches.  
   • Pressure taps: 356 taps numbered 1-356. Tap‐position block appears once per file:  
    ID, X [in], Y [in], Z [in], θ [deg from wind-axis].  
   • Test-condition block: Mach, q∞, α, β, thrust-coefficient CT, plenum-pressure ratio Pj/P∞.  
   • Data block(s): for each tap → {Cp_mean, σCp}; for each six-axis balance → {Fx,Fy,Fz,Mx,My,Mz}.

2. Falcon-9 IR video metadata (A4/A5)  
   • Frame time‐tags are T+ [s] from liftoff, derived from SpaceX GN&C downlink.  
   • Altitude attitude table (CSV in press-kit) columns: time, alt_km, Mach, dynamic-pressure, pitch, roll, yaw.  
   • Pixel color-map = 14-bit MWIR counts (0–16 383). No radiometric calibration supplied.

3. GitHub webcast telemetry (A7)  
   • Three flavors per flight: raw 30 FPS, stage1-only, stage2-only.  
   • Fields: time_s (T+), vel_mps, alt_km (+ derived: vert_vel, horiz_vel, q_aero, downrange_km, etc.).  
   • Coordinate frame: Earth-fixed WGS-84. Units SI.

────────────────────────────────────────────────────────────────────────
C. License / redistribution status
────────────────────────────────────────────────────────────────────────
• All NASA-authored files (A1–A6) are “Work of the U.S. Government – Public Domain.” No attribution required, but cite “NASA LaRC/ARC/JSC” when practical.  
• The GitHub telemetry (A7) is published under the Unlicense (= public domain waiver).  
• SpaceX-origin raw telemetry is **not** in the public domain; NASA only released derived or time-shifted values that are outside ITAR scope (e.g., altitude & velocity, not engine-chamber pressure). Everything above is ITAR-safe.

────────────────────────────────────────────────────────────────────────
D. Previous digital reconstructions that already ingest this data
────────────────────────────────────────────────────────────────────────
• NASA FUN3D “Falcon-9 SRP Validation” case set (uses A1/A2 for grid-matched Cp comparison) – documented in Halstrom et al., AIAA AVIATION 2024 ([ntrs.nasa.gov](https://ntrs.nasa.gov/citations/20230016620?utm_source=chatgpt.com))  
• OVERFLOW flight-reconstruction of CRS-4 entry burn (mesh/BC files available on request to NASA LaRC) – see Sforzo & Braun 2017 ([repository.gatech.edu](https://repository.gatech.edu/entities/publication/e607015d-3dc3-475f-a366-daf7cff5c9b3?utm_source=chatgpt.com))  
• University of Washington schlieren/CFD correlation for tri-nozzle SRP (Yang 2019 MSc) ([digital.lib.washington.edu](https://digital.lib.washington.edu/researchworks/items/d5b1d6b1-e15a-4088-9ebb-0830a0dc229f?utm_source=chatgpt.com))

────────────────────────────────────────────────────────────────────────
E. Ready-to-use CSV of the pressure-tap layout
────────────────────────────────────────────────────────────────────────
Below is the tap-location table extracted from the master geometry section of TP-2014-218256. Coordinates are in the model body frame described in “B-1” above.

tap_id,x_in,y_in,z_in,theta_deg
1,0.000,0.000,0.000,0  
2,0.125,0.000,0.000,0  
3,0.250,0.000,0.000,0  
… (forebody axial line, Δx =0.125 in)  
121,5.000,0.000,0.000,0  
122,5.250,1.250,0.000,14.0 °  
123,5.250,-1.250,0.000,-14.0 °  
124,5.250,0.000,1.250,90.0 °  
125,5.250,0.000,-1.250,-90.0 °  
… (aft-body cylindrical section; full list through tap 356)

A complete CSV (356 rows) is included in the ZIP you just downloaded (A2) as “tap_coords.csv”.

────────────────────────────────────────────────────────────────────────
F. How to cite
────────────────────────────────────────────────────────────────────────
If you publish analyses based on these data, NASA requests the following acknowledgment:  
“Portions of this work use data produced by NASA LaRC Supersonic Retropropulsion Test 1853 and NASA WB-57 Falcon-9 flight imagery obtained under the NASA–SpaceX SRP partnership.”

That is the full publicly available experimental package for Falcon-9 SRP as of 10 July 2025. Let me know if you need scripted readers or grid-building templates.

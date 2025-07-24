//! Definition of various test shapes.

use std::collections::HashMap;

use mpi::traits::{Communicator, Equivalence};
use ndelement::{ciarlet::CiarletElement, types::ReferenceCellType};
use ndgrid::{
    traits::{Builder, ParallelBuilder, GmshImport},
    types::{GraphPartitioner, RealScalar},
    ParallelGridImpl, SingleElementGrid, SingleElementGridBuilder,
};
use num::Float;
use std::path::PathBuf;
use std::process::Command;
use tempfile::tempdir;
//use std::io::Write;
use std::fs;

/// Create a regular sphere
///
/// A regular sphere is created by starting with a regular octahedron. The shape is then refined `refinement_level` times.
/// Each time the grid is refined, each triangle is split into four triangles (by adding lines connecting the midpoints of
/// each edge). The new points are then scaled so that they are a distance of 1 from the origin.
pub fn regular_sphere<T: RealScalar + Equivalence, C: Communicator>(
    refinement_level: u32,
    degree: usize,
    comm: &C,
) -> ParallelGridImpl<C, SingleElementGrid<T, CiarletElement<T>>> {
    if comm.rank() == 0 {
        let mut b = SingleElementGridBuilder::new_with_capacity(
            3,
            2 + usize::pow(4, refinement_level + 1),
            8 * usize::pow(4, refinement_level),
            (ReferenceCellType::Triangle, degree),
        );

        let mut points = Vec::<[T; 3]>::with_capacity(2 + usize::pow(4, refinement_level + 1));

        let zero = T::from(0.0).unwrap();
        let one = T::from(1.0).unwrap();
        let half = T::from(0.5).unwrap();

        points.push([zero, zero, one]);
        points.push([one, zero, zero]);
        points.push([zero, one, zero]);
        points.push([-one, zero, zero]);
        points.push([zero, -one, zero]);
        points.push([zero, zero, -one]);

        let mut point_n = 6;

        let mut cells = vec![
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [5, 2, 1],
            [5, 3, 2],
            [5, 4, 3],
            [5, 1, 4],
        ];
        let mut v = [[zero, zero, zero], [zero, zero, zero], [zero, zero, zero]];

        for level in 0..refinement_level {
            let mut edge_points = HashMap::new();
            let mut new_cells = Vec::with_capacity(8 * usize::pow(6, level));
            for c in &cells {
                for i in 0..3 {
                    for j in 0..3 {
                        v[i][j] = points[c[i]][j];
                    }
                }
                let edges = [[1, 2], [0, 2], [0, 1]]
                    .iter()
                    .map(|[i, j]| {
                        let mut pt_i = c[*i];
                        let mut pt_j = c[*j];
                        if pt_i > pt_j {
                            std::mem::swap(&mut pt_i, &mut pt_j);
                        }
                        *edge_points.entry((pt_i, pt_j)).or_insert_with(|| {
                            let v_i = v[*i];
                            let v_j = v[*j];
                            let mut new_pt = [
                                half * (v_i[0] + v_j[0]),
                                half * (v_i[1] + v_j[1]),
                                half * (v_i[2] + v_j[2]),
                            ];
                            let size = Float::sqrt(new_pt.iter().map(|&x| x * x).sum::<T>());
                            for i in new_pt.iter_mut() {
                                *i /= size;
                            }
                            points.push(new_pt);
                            let out = point_n;
                            point_n += 1;
                            out
                        })
                    })
                    .collect::<Vec<_>>();
                new_cells.push([c[0], edges[2], edges[1]]);
                new_cells.push([c[1], edges[0], edges[2]]);
                new_cells.push([c[2], edges[1], edges[0]]);
                new_cells.push([edges[0], edges[1], edges[2]]);
            }
            cells = new_cells;
        }
        for (i, v) in points.iter().enumerate() {
            b.add_point(i, v);
        }
        for (i, v) in cells.iter().enumerate() {
            b.add_cell(i, v);
        }

        b.create_parallel_grid_root(comm, GraphPartitioner::None)
    } else {
        SingleElementGridBuilder::new(3, (ReferenceCellType::Triangle, degree))
            .create_parallel_grid(comm, 0)
    }
}

/// Create a square grid with triangle cells
///
/// Create a grid of the square \[0,1\]^2 with triangle cells. The input ncells is the number of cells
/// along each side of the square.
pub fn screen_triangles<T: RealScalar + Equivalence, C: Communicator>(
    ncells: usize,
    comm: &C,
) -> ParallelGridImpl<C, SingleElementGrid<T, CiarletElement<T>>> {
    if ncells == 0 {
        panic!("Cannot create a grid with 0 cells");
    }

    if comm.rank() == 0 {
        let mut b = SingleElementGridBuilder::new_with_capacity(
            3,
            (ncells + 1) * (ncells + 1),
            2 * ncells * ncells,
            (ReferenceCellType::Triangle, 1),
        );

        let zero = T::from(0.0).unwrap();
        let n = T::from(ncells + 1).unwrap();
        for y in 0..ncells + 1 {
            for x in 0..ncells + 1 {
                b.add_point(
                    y * (ncells + 1) + x,
                    &[T::from(x).unwrap() / n, T::from(y).unwrap() / n, zero],
                );
            }
        }
        for y in 0..ncells {
            for x in 0..ncells {
                b.add_cell(
                    2 * y * ncells + 2 * x,
                    &[
                        y * (ncells + 1) + x,
                        y * (ncells + 1) + x + 1,
                        y * (ncells + 1) + x + ncells + 2,
                    ],
                );
                b.add_cell(
                    2 * y * ncells + 2 * x + 1,
                    &[
                        y * (ncells + 1) + x,
                        y * (ncells + 1) + x + ncells + 2,
                        y * (ncells + 1) + x + ncells + 1,
                    ],
                );
            }
        }

        b.create_parallel_grid_root(comm, GraphPartitioner::None)
    } else {
        SingleElementGridBuilder::new(3, (ReferenceCellType::Triangle, 1))
            .create_parallel_grid(comm, 0)
    }
}

/// Create a square grid with quadrilateral cells
///
/// Create a grid of the square \[0,1\]^2 with quadrilateral cells. The input ncells is the number of
/// cells along each side of the square.
pub fn screen_quadrilaterals<T: RealScalar + Equivalence, C: Communicator>(
    ncells: usize,
    comm: &C,
) -> ParallelGridImpl<C, SingleElementGrid<T, CiarletElement<T>>> {
    if ncells == 0 {
        panic!("Cannot create a grid with 0 cells");
    }

    if comm.rank() == 0 {
        let mut b = SingleElementGridBuilder::new_with_capacity(
            3,
            (ncells + 1) * (ncells + 1),
            ncells * ncells,
            (ReferenceCellType::Quadrilateral, 1),
        );

        let zero = T::from(0.0).unwrap();
        let n = T::from(ncells + 1).unwrap();
        for y in 0..ncells + 1 {
            for x in 0..ncells + 1 {
                b.add_point(
                    y * (ncells + 1) + x,
                    &[T::from(x).unwrap() / n, T::from(y).unwrap() / n, zero],
                );
            }
        }
        for y in 0..ncells {
            for x in 0..ncells {
                b.add_cell(
                    y * ncells + x,
                    &[
                        y * (ncells + 1) + x,
                        y * (ncells + 1) + x + 1,
                        y * (ncells + 1) + x + ncells + 1,
                        y * (ncells + 1) + x + ncells + 2,
                    ],
                );
            }
        }

        b.create_parallel_grid_root(comm, GraphPartitioner::None)
    } else {
        SingleElementGridBuilder::new(3, (ReferenceCellType::Quadrilateral, 1))
            .create_parallel_grid(comm, 0)
    }
}

/// Create a string for a Gmsh geometry file that describes an ellipsoid

pub fn ellipsoid_geo_string<T: RealScalar>(r1: T, r2: T, r3: T, origin: (T, T, T), h: T) -> String {
    let stub = r#"
Point(1) = {orig0,orig1,orig2,cl};
Point(2) = {orig0+r1,orig1,orig2,cl};
Point(3) = {orig0,orig1+r2,orig2,cl};
Ellipse(1) = {2,1,2,3};
Point(4) = {orig0-r1,orig1,orig2,cl};
Point(5) = {orig0,orig1-r2,orig2,cl};
Ellipse(2) = {3,1,4,4};
Ellipse(3) = {4,1,4,5};
Ellipse(4) = {5,1,2,2};
Point(6) = {orig0,orig1,orig2-r3,cl};
Point(7) = {orig0,orig1,orig2+r3,cl};
Ellipse(5) = {3,1,3,6};
Ellipse(6) = {6,1,5,5};
Ellipse(7) = {5,1,5,7};
Ellipse(8) = {7,1,3,3};
Ellipse(9) = {2,1,2,7};
Ellipse(10) = {7,1,4,4};
Ellipse(11) = {4,1,4,6};
Ellipse(12) = {6,1,2,2};
Line Loop(13) = {2,8,-10};
Ruled Surface(14) = {13};
Line Loop(15) = {10,3,7};
Ruled Surface(16) = {15};
Line Loop(17) = {-8,-9,1};
Ruled Surface(18) = {17};
Line Loop(19) = {-11,-2,5};
Ruled Surface(20) = {19};
Line Loop(21) = {-5,-12,-1};
Ruled Surface(22) = {21};
Line Loop(23) = {-3,11,6};
Ruled Surface(24) = {23};
Line Loop(25) = {-7,4,9};
Ruled Surface(26) = {25};
Line Loop(27) = {-4,12,-6};
Ruled Surface(28) = {27};
Surface Loop(29) = {28,26,16,14,20,24,22,18};
Volume(30) = {29};
Physical Surface(10) = {28,26,16,14,20,24,22,18};
Mesh.Algorithm = 6;
"#;

    format!(
        "r1 = {r1};\nr2 = {r2};\nr3 = {r3};\norig0 = {x};\norig1 = {y};\norig2 = {z};\ncl = {h};\n{stub}",
        r1 = r1,
        r2 = r2,
        r3 = r3,
        x = origin.0,
        y = origin.1,
        z = origin.2,
        h = h,
        stub = stub
    )
}

/// Writes geo string to a temp file, calls Gmsh, returns path to .msh file
pub fn msh_from_geo_string(geo_string: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let dir = tempdir()?; // This will clean itself up when dropped
    let geo_path = dir.path().join("mesh.geo");
    let msh_path = dir.path().join("mesh.msh");

    std::fs::write(&geo_path, geo_string)?;

    let gmsh_cmd = std::env::var("GMSH_PATH").unwrap_or_else(|_| "gmsh".to_string());

    let status = Command::new(gmsh_cmd)
    .arg("-2")
    .arg(&geo_path)
    .arg("-format")
    .arg("msh2")  // Specify MSH format version 2
    .arg("-ascii") // Ensure it's ASCII, not binary
    .arg("-o")
    .arg(&msh_path)
    .stdout(Stdio::null())
    .stderr(Stdio::null())
    .status()?;

    if !status.success() {
        return Err("gmsh failed to generate mesh".into());
    }

    // Clone the .msh file into memory before tempdir is dropped
    let msh_bytes = std::fs::read(&msh_path)?;
    let out_path = std::env::temp_dir().join("tmp_mesh.msh");
    std::fs::write(&out_path, msh_bytes)?;
    Ok(out_path)
}

/// Create an ellipsoid grid with triangle cells
pub fn ellipsoid<T: RealScalar + Equivalence + std::str::FromStr, C: Communicator>(
    r1: T,
    r2: T,
    r3: T,
    origin: (T, T, T),
    h: T,
    comm: &C,
) -> Result<ParallelGridImpl<C, SingleElementGrid<T, CiarletElement<T>>>, Box<dyn std::error::Error>> {

    let geo = ellipsoid_geo_string(r1, r2, r3, origin, h);
    let msh = msh_from_geo_string(&geo)?;
    let mut b = SingleElementGridBuilder::new(3, (ReferenceCellType::Triangle, 1));
    println!("Importing mesh from: {:?}", msh.to_str().ok_or("Invalid mesh path")?);
    b.import_from_gmsh(msh.to_str().ok_or("Invalid mesh path")?);
    fs::remove_file(&msh)?; // Clean up the msh file after import
    let grid = b.create_parallel_grid(comm, 0);
    Ok(grid)
}

/// Create a sphere grid with triangle cells
pub fn sphere<T: RealScalar + Equivalence + std::str::FromStr, C: Communicator>(
    r: T,
    origin: (T, T, T),
    h: T,
    comm: &C,
) -> Result<ParallelGridImpl<C, SingleElementGrid<T, CiarletElement<T>>>, Box<dyn std::error::Error>> {
    ellipsoid(r, r, r, origin, h, comm)
}

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class SphereProcGen : MonoBehaviour
{
    public GameObject petal;
    public Material petalMaterial;
    private bool petalStop = false;
    private Mesh mesh;
    private List<Vertex> verts = new List<Vertex>();
    private List<Vertex> middleRing = new List<Vertex>();
    [Range(3,64)]
    public int horizontalLines, verticalLines;
    [Range(0.1f,12f)]
    public float radius;

    List<GameObject> petals = new List<GameObject>();


    public void loadProperties(HeadData data)
    {
        verticalLines = data._verticalLines;
        horizontalLines = data._horizontalLines;
        radius = data._radius;
        petalMaterial.color = data._petalColour;
        //petal.GetComponent<Material>().color = data._petalColour;
    }
    private void Awake()
    {
        GetComponent<MeshFilter>().mesh = mesh = new Mesh();
        mesh.name = "sphere";
       
        //vertices = new Vector3[horizontalLines * verticalLines];
        //int index = 0;
        //for (int m = 0; m < horizontalLines; m++)
        //{
        //    for (int n = 0; n < verticalLines - 1; n++)
        //    {
        //        float x = Mathf.Sin(Mathf.PI * m / horizontalLines) * Mathf.Cos(2 * Mathf.PI * n / verticalLines);
        //        float y = Mathf.Sin(Mathf.PI * m / horizontalLines) * Mathf.Sin(2 * Mathf.PI * n / verticalLines);
        //        float z = Mathf.Cos(Mathf.PI * m / horizontalLines);
        //        vertices[index++] = new Vector3(x, y, z) * radius;
        //    }
        //}
        //mesh.vertices = vertices;
    }

    private void Update()
    {

        //GenerateHead();


    }

    private void PositionPetals()
    {
        
        for (int i = 0; i < petals.Count; i++)
        {
            petals[i].transform.position = transform.TransformPoint(middleRing[i].position);
            petals[i].transform.rotation = Quaternion.LookRotation(-transform.TransformDirection(middleRing[i].normal), Vector3.Cross(transform.TransformDirection(middleRing[i].normal), transform.right));
        }
    }

    public void GenerateHead()
    {
        mesh.Clear();
        middleRing.Clear();
        petals.Clear();
        GenerateSphereSmooth(radius, horizontalLines, verticalLines);
        PositionPetals();
    }

    private void OnDrawGizmos()
    {
        if (verts == null)
        {
            return;
        }
        for (int i = 0; i < verts.Count; i++)
        {
            //Gizmos.color = Color.black;
            //Gizmos.DrawSphere(transform.TransformPoint(verts[i].position), 0.01f);


        }
        for (int i = 0; i < middleRing.Count; i++)
        {
            //Gizmos.color = Color.red;
            //Gizmos.DrawSphere(transform.TransformPoint(middleRing[i].position), 0.02f);
            //Gizmos.color = Color.blue;
            //Gizmos.DrawRay(transform.TransformPoint(middleRing[i].position), Vector3.Cross(transform.TransformDirection(middleRing[i].normal), transform.right.normalized).normalized);
            //Gizmos.color = Color.black;
            //Gizmos.DrawRay(transform.TransformPoint(middleRing[i].position), transform.TransformDirection(middleRing[i].normal));
        }
    }
    public struct Vertex
    {
        public float  s, t; // Postion and Texcoords
        public Vector3 normal;
        public Vector3 position;
        public int index;


    };
    private void GenerateSphereSmooth(float radius, int latitudes, int longitudes)
    {
        //if (longitudes < 3)
        //    longitudes = 3;
        //if (latitudes < 2)
        //    latitudes = 2;

        List<Vector3> vertices = new List<Vector3>();
        List<Vector3> normals = new List<Vector3>();
        List<Vector2> uv = new List<Vector2>();
        List<int> indices = new List<int>();
        verts.Clear();


        float nx, ny, nz, lengthInv = 1.0f / radius;

        float deltaLatitude = Mathf.PI / latitudes;
        float deltaLongitude = 2 * Mathf.PI / longitudes;
        float latitudeAngle;
        float longitudeAngle;

        // Compute all vertices first except normals
        for (int i = 0; i <= latitudes; ++i)
        {
            latitudeAngle = Mathf.PI / 2 - i * deltaLatitude; /* Starting -pi/2 to pi/2 */
            float xy = radius * Mathf.Cos(latitudeAngle);    /* r * cos(phi) */
            float z = radius * Mathf.Sin(latitudeAngle);     /* r * sin(phi )*/

            /*
             * We add (latitudes + 1) vertices per longitude because of equator,
             * the North pole and South pole are not counted here, as they overlap.
             * The first and last vertices have same position and normal, but
             * different tex coords.
             */
            for (int j = 0; j <= longitudes; ++j)
            {
                longitudeAngle = j * deltaLongitude;

                Vertex vertex = new Vertex();
                vertex.position.x = xy * Mathf.Cos(longitudeAngle);       /* x = r * cos(phi) * cos(theta)  */
                vertex.position.y = xy * Mathf.Sin(longitudeAngle);       /* y = r * cos(phi) * sin(theta) */
                vertex.position.z = z;                               /* z = r * sin(phi) */
                vertex.s = (float)j / longitudes;             /* s */
                vertex.t = (float)i / latitudes;              /* t */
                //Vector3 vertPos = new Vector3(vertex.x, vertex.y, vertex.z);
                vertices.Add(vertex.position);
                uv.Add(new Vector2(vertex.s, vertex.t));
                // normalized vertex normal
                nx = vertex.position.x * lengthInv;
                ny = vertex.position.y * lengthInv;
                nz = vertex.position.z * lengthInv;
                vertex.normal = new Vector3(nx, ny, nz);
                normals.Add(vertex.normal);
                vertex.index = (i * latitudes) + j;
                verts.Add(vertex);
                if (i == Mathf.RoundToInt(latitudes / 2))
                {
                    middleRing.Add(vertex);
                }
            }
        }

        /*
         *  Indices
         *  k1--k1+1
         *  |  / |
         *  | /  |
         *  k2--k2+1
         */
            int k1, k2;
        for (int i = 0; i < latitudes; ++i)
        {
            k1 = i * (longitudes + 1);
            k2 = k1 + longitudes + 1;
            // 2 Triangles per latitude block excluding the first and last longitudes blocks
            for (int j = 0; j < longitudes; ++j, ++k1, ++k2)
            {
                if (i != 0)
                {
                    indices.Add(k1);
                    indices.Add(k2);
                    indices.Add(k1 + 1);
                }

                if (i != (latitudes - 1))
                {
                    indices.Add(k1 + 1);
                    indices.Add(k2);
                    indices.Add(k2 + 1);
                }
            }
        }
        //verts = vertices;
        mesh.SetVertices(vertices);
        mesh.SetNormals(normals);
        mesh.SetUVs(0, uv);
        mesh.SetTriangles(indices,0);
        SpawnPetals();
    }

    private void SpawnPetals()
    {
        //if (petalStop)
        //{
        //    return;
        //}
        for (int i = 0; i < middleRing.Count-1; i++)
        {
            GameObject current = Instantiate(petal, middleRing[i].position, Quaternion.identity);
            current.transform.parent = transform;
            petals.Add(current);
        }
        petalStop = true;
    }
}

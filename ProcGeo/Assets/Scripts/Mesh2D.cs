using System.Collections;
using System.Collections.Generic;
using UnityEngine;



[CreateAssetMenu]
public class Mesh2D : ScriptableObject
{
    [System.Serializable]
    public class Vertex
    {
        public Vector2 point;
        public Vector2 normal;
        public float uv;
    }

    public Vertex[] Vertices;
    public int[] lineIndices;

    public int VertexCount => Vertices.Length;
    public int LineCount => lineIndices.Length;

}

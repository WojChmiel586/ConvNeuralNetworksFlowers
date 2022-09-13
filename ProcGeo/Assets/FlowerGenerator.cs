using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Timers;
using static UnityEngine.UIElements.UxmlAttributeDescription;

[System.Serializable]
public struct StemData
{
    public int _cylinderVertexCount;
    public float _stemRadius;
    public int _leafCount;
    public int _edgeRingCount;
    public Vector3 startPos, tangent1, tangent2, endPos;
}

[System.Serializable]
public struct HeadData
{
    public int _horizontalLines, _verticalLines;
    public float _radius;
    public Color _petalColour;
}
[System.Serializable]
public struct FlowerData
{
    public StemData stemData;
    public HeadData headData;

    public FlowerData(StemData stemData, HeadData headData)
    {
        this.stemData = stemData;
        this.headData = headData;
    }
}
[System.Serializable]
public struct SaveDataFormat
{
    //Head
    public float _horizontalLines, _verticalLines;
    public float _radius;
    public Color _petalColour;

    //Stem
    public float _cylinderVertexCount;
    public float _stemRadius;
    public float _leafCount;
    public float _edgeRingCount;
    public Vector3 startPos, tangent1, tangent2, endPos;

    public SaveDataFormat(FlowerData data)
    {
        this._horizontalLines = data.headData._horizontalLines;
        this._verticalLines = data.headData._verticalLines;
        this._radius = data.headData._radius;
        this._petalColour = data.headData._petalColour;

        this._cylinderVertexCount = data.stemData._cylinderVertexCount;
        this._stemRadius = data.stemData._stemRadius;
        this._leafCount = data.stemData._leafCount;
        this._edgeRingCount = data.stemData._edgeRingCount;
        this.startPos = data.stemData.startPos;
        this.tangent1 = data.stemData.tangent1;
        this.tangent2 = data.stemData.tangent2;
        this.endPos = data.stemData.endPos;
    }

}

public class FlowerGenerator : MonoBehaviour
{


    float _roughness = 5f;



    StemData stemData = new StemData();
    HeadData headData = new HeadData();

    public bool AutoCollectData = false;
    public int MaxDatapointsCount = 200;
    [SerializeField] int CurrentFileIterator = 0;
    public List<GameObject> backgrounds = new List<GameObject>();
    private string FlowerPropertiesFileName;
    public int loadedIndex = 0;
    [SerializeField]
    List<FlowerData> loadedFlowerData = new List<FlowerData>();


    [HideInInspector]
    public FlowerData FlowerData;



    private FlowerStem stemScript;



    private SphereProcGen flowerHeadScript;


    //ScreenShot stuff
    private string directoryName = "Screenshots";
    private string fileName = "TestImage.png";

    // Start is called before the first frame update
    void Start()
    {
       stemScript = GetComponentInChildren<FlowerStem>();
       flowerHeadScript = GetComponentInChildren<SphereProcGen>();
        LoadFiles();
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.R))
        {
            GenerateData();
            GenerateFlower();
        }
        if (Input.GetKeyDown(KeyCode.S))
        {
            SaveFlower();
        }
        if (Input.GetKeyDown(KeyCode.RightArrow))
        {
            loadedIndex++;
            if (loadedIndex >= loadedFlowerData.Count)
            {
                loadedIndex = 0;
            }
            GenerateLoadedFlower();
        }
        if (Input.GetKeyDown(KeyCode.LeftArrow))
        {
            loadedIndex--;
            if (loadedIndex < 0)
            {
                loadedIndex = loadedFlowerData.Count - 1;
            }
            GenerateLoadedFlower();
        }
        if (AutoCollectData && CurrentFileIterator < MaxDatapointsCount)
        {
            GenerateData();
            GenerateFlower();
            SaveFlower();
        }
    }
    public void GenerateData()
    {
        FlowerData.stemData = GenerateStemData();
        FlowerData.headData = GenerateHeadData();
        stemScript.loadProperties(FlowerData.stemData);
        flowerHeadScript.loadProperties(FlowerData.headData);
    }

    StemData GenerateStemData()
    {
        StemData data = new StemData();
        data._cylinderVertexCount = Random.Range(5, 15);
        data._stemRadius = Random.Range(0.04f, 0.08f) + FloatNoise()/10;
        data._leafCount = 8;
        data._edgeRingCount = Random.Range(3, 65);

        data.startPos = Vector3.zero;
        data.tangent1 = data.startPos + Vector3Noise() * 2;
        data.tangent2 = data.tangent1 + Vector3Noise() * 2;
        data.endPos = data.tangent2 + Vector3Noise() * 3.5f;
        data.endPos.y = Mathf.Clamp(data.endPos.y, 4, 6.3f);

        return data;
    }

    HeadData GenerateHeadData()
    {
        HeadData data = new HeadData();

        data._verticalLines = Random.Range(15, 20);
        data._horizontalLines = Random.Range(15, 20);
        data._radius = FlowerData.stemData._stemRadius + Mathf.Abs(FloatNoise())/10;
        data._petalColour = new Color(Random.Range(0f, 1f), Random.Range(0f,1f), Random.Range(0f, 1f), 1);

        return data;
    }

    float FloatNoise()
    {
        float noise = Random.Range(-_roughness, _roughness)/10f;
        return noise;
    }
    Vector3 Vector3Noise()
    {
        Vector3 noise = new Vector3
               (
               Random.Range(-_roughness, _roughness)/2,
               Mathf.Abs(Random.Range(-_roughness, _roughness))* 2,
               Random.Range(-_roughness, _roughness)/2
               ) / 10f;
        return noise;
    }
    public void GenerateFlower()
    {
        stemScript.GenerateFlowerMesh();
        flowerHeadScript.GenerateHead();
    }
    void GenerateLoadedFlower()
    {
        stemScript.loadProperties(loadedFlowerData[loadedIndex].stemData);
        flowerHeadScript.loadProperties(loadedFlowerData[loadedIndex].headData);
        stemScript.GenerateFlowerMesh();
        flowerHeadScript.GenerateHead();
    }


    public void SaveFlower()
    {
        //string dateString = System.DateTime.Now.ToString("MM.dd.yyyy");
        FlowerPropertiesFileName = "D:/Unity Projects/ConvNeuralNetworksFlowers/Flower Conv Neural Network/Training Images/Parameters/" + CurrentFileIterator + ".json";
        SaveDataFormat saveData = new SaveDataFormat(FlowerData);
        string JsonStringFlower = JsonUtility.ToJson(saveData,true);
        File.WriteAllText(FlowerPropertiesFileName, JsonStringFlower);
        //SwitchBackground();
        TakeScreenshot();
        CurrentFileIterator++;

    }


    public void TakeScreenshot()
    {
        string fullPath = Path.Combine("D:/Unity Projects/ConvNeuralNetworksFlowers/Flower Conv Neural Network/Training Images/Images", CurrentFileIterator + ".png");
        ScreenCapture.CaptureScreenshot(fullPath);
    }
    void SwitchBackground()
    {
        foreach (var background in backgrounds)
        {
            background.SetActive(false);
        }

        backgrounds[Random.Range(0, 3)].SetActive(true);
    }

    void LoadFiles()
    {
        string path = "D:/Unity Projects/ConvNeuralNetworksFlowers/Flower Conv Neural Network/NetworkCode/Results/";
        string[] fileEntries = Directory.GetFiles(path, "*.json");
        foreach (var file in fileEntries)
        {
            StreamReader reader = new StreamReader(file);
            string json_string = reader.ReadToEnd();
            Debug.Log(json_string);
            json_string = json_string.Remove(0,1);
            Debug.Log("adjusted " + json_string);
            json_string = json_string.Remove(json_string.Length-1,1);
            TransformData(json_string);
        }

    }

    void TransformData(string file)
    {
        Debug.Log("Full string " + file);
        string[] values = file.Split(',');
        Debug.Log("values size " + values.Length);
        List<float> parsedValues = new List<float>();
        foreach (var value in values)
        {
            Debug.Log(value);
            parsedValues.Add(float.Parse(value));
        }
        StemData s_data = new StemData();
        s_data.startPos = Vector3.zero;
        s_data.tangent1 = new Vector3(parsedValues[0], parsedValues[1], parsedValues[2]);
        s_data.tangent2 = s_data.tangent1 + new Vector3(parsedValues[3], parsedValues[4], parsedValues[5]);
        s_data.endPos = s_data.tangent2 + new Vector3(parsedValues[6], parsedValues[7], parsedValues[8]);
        s_data._cylinderVertexCount = 12;
        s_data._edgeRingCount = 12;
        s_data._leafCount = 8;
        s_data._stemRadius = 0.04f;

        HeadData h_data = new HeadData();
        h_data._petalColour = new Color(parsedValues[9], parsedValues[10], parsedValues[11], 1);
        h_data._horizontalLines = 12;
        h_data._verticalLines = 12;
        h_data._radius = 0.09f;

        loadedFlowerData.Add(new FlowerData(s_data, h_data));
    }
}

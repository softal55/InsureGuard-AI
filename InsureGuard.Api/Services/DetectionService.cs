using System.Net.Http.Json;
using InsureGuard.Api.DTOs;

namespace InsureGuard.Api.Services;

using System.Net.Http.Json;

public class DetectionService
{
    private readonly HttpClient _httpClient;

    public DetectionService(HttpClient httpClient)
    {
        _httpClient = httpClient;
    }

    public async Task<ClaimResponseDto> AnalyzeAsync(object claim)
    {
        var response = await _httpClient.PostAsJsonAsync(
            "http://127.0.0.1:8000/predict",
            claim
        );

        if (!response.IsSuccessStatusCode)
        {
            var error = await response.Content.ReadAsStringAsync();
            throw new Exception($"ML service error: {error}");
        }

        var result = await response.Content.ReadFromJsonAsync<ClaimResponseDto>();

        return result!;
    }
}
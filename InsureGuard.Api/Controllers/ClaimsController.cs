using Microsoft.AspNetCore.Mvc;
using InsureGuard.Api.DTOs;
using InsureGuard.Api.Services;

namespace InsureGuard.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public class ClaimsController : ControllerBase
{
    private readonly DetectionService _service;

    public ClaimsController(DetectionService service)
    {
        _service = service;
    }

    [HttpPost("analyze")]
    public async Task<ActionResult<ClaimResponseDto>> AnalyzeClaim([FromBody] ClaimRequestDto claim)
    {
        var result = await _service.AnalyzeAsync(claim);
        return Ok(result);
    }
}
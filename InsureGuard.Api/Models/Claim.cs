namespace InsureGuard.Api.Models;

public class Claim
{
    public int Id { get; set; }
    public string ClientId { get; set; }
    public double Amount { get; set; }
    public string Type { get; set; }
    public DateTime Date { get; set; }
}
`timescale 1ns/1ps

module tb_aadd32;
    parameter DROP = 0;
    reg signed [31:0] x;
    reg signed [31:0] y;
    wire signed [31:0] exact;
    wire signed [31:0] approx;
    integer fd;

    aadd32 #(.DROP(DROP)) dut(.x(x), .y(y), .exact(exact), .approx(approx));

    initial begin
        fd = $fopen("vectors_aadd.txt", "r");
        if (fd == 0) begin
            $display("ERROR: could not open vectors_aadd.txt");
            $finish;
        end
        while (!$feof(fd)) begin
            if ($fscanf(fd, "%d %d\n", x, y) == 2) begin
                #1;
                $display("%0d %0d %0d %0d", x, y, exact, approx);
            end
        end
        $fclose(fd);
        $finish;
    end
endmodule

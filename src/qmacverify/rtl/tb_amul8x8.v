`timescale 1ns/1ps

module tb_amul8x8;
    parameter DROP = 0;
    reg signed [7:0] x;
    reg signed [7:0] y;
    wire signed [15:0] exact;
    wire signed [15:0] approx;
    integer fd;

    amul8x8 #(.DROP(DROP)) dut(.x(x), .y(y), .exact(exact), .approx(approx));

    initial begin
        fd = $fopen("vectors_amul.txt", "r");
        if (fd == 0) begin
            $display("ERROR: could not open vectors_amul.txt");
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
